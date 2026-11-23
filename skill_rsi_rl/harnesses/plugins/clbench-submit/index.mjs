import {
  appendFileSync, closeSync, constants, existsSync, fstatSync, lstatSync, mkdirSync, openSync,
  readFileSync, readSync, realpathSync, renameSync, rmSync, writeFileSync,
} from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { registerContextFind, visibleSources } from './context-find.mjs';
import { isContextEditTool, registerContextEdits } from './context-edit.mjs';
import { projectContextView } from './context-view.mjs';

export const name = 'clbench-submit';
export const inject = ['tools', 'systemPrompt', 'llm'];

function physical(path) {
  let ancestor = resolve(path);
  const suffix = [];
  while (!existsSync(ancestor)) {
    const parent = dirname(ancestor);
    if (parent === ancestor) throw new Error(`Cannot resolve path ${path}`);
    suffix.unshift(relative(parent, ancestor));
    ancestor = parent;
  }
  return join(realpathSync(ancestor), ...suffix);
}

const MEMORY_TOOLS = new Set(['read', 'write', 'edit']);
const MAX_MEMORY_BYTES = 4 * 1024 * 1024;

function inside(root, path) {
  const part = relative(root, path);
  return part !== '..' && !part.startsWith(`..${sep}`) && !isAbsolute(part);
}

function recorder(config) {
  const directory = config.captureDirectory;
  const collector = randomUUID();
  const counts = {};
  const errors = {};
  const activeCalls = new Set();
  const run = (source, fn) => {
    try {
      return fn();
    } catch (error) {
      const key = `${source}:${error?.code ?? error?.name ?? 'Error'}`;
      errors[key] = (errors[key] ?? 0) + 1;
      return undefined;
    }
  };
  const save = (path, content) => {
    const pending = `${path}.${randomUUID()}.tmp`;
    try {
      writeFileSync(pending, content, { flag: 'wx', mode: 0o600 });
      renameSync(pending, path);
    } finally {
      rmSync(pending, { force: true });
    }
  };
  const append = (path, row) => appendFileSync(path, `${JSON.stringify(row)}\n`, { mode: 0o600 });
  const event = (source, kind, sessionId, fields) => {
    if (!directory) return;
    const seq = counts[source] = (counts[source] ?? 0) + 1;
    run(source, () => append(join(directory, `${source}.jsonl`), {
      schema_version: 1, source_seq: seq, collector_id: collector, timestamp: new Date().toISOString(),
      kind, session_id: sessionId ?? null, runtime_id: config.runtimeId, ...fields,
    }));
  };
  const health = disposed => directory && run('plugin_health', () => save(join(directory, 'plugin_health.json'),
    JSON.stringify({
      collector_id: collector, disposed, counts, errors, open_calls: [...activeCalls],
      memory_max_snapshot_bytes: MAX_MEMORY_BYTES,
    })));
  for (const sub of ['objects', 'responses']) {
    if (directory) run(sub, () => mkdirSync(join(directory, sub), { recursive: true, mode: 0o700 }));
  }
  health(false);
  return { run, save, event, health, append, activeCalls };
}

function contentObject(capture, directory, content, representation) {
  if (content.length > MAX_MEMORY_BYTES) return { status: 'oversize', bytes: content.length, representation };
  const sha256 = createHash('sha256').update(content).digest('hex');
  const ref = join('objects', sha256);
  const file = join(directory, ref);
  if (!existsSync(file)) capture.save(file, content);
  return { status: 'captured', sha256, path: `capture/${ref}`, bytes: content.length, representation };
}

function snapshot(capture, directory, root, requested) {
  const outcome = capture.run('memory_snapshot', () => {
    const path = physical(requested);
    if (path === root || !inside(root, path)) return { status: 'outside_scope' };
    let fd;
    try {
      fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    } catch (error) {
      if (error?.code === 'ENOENT') return { status: 'absent' };
      throw error;
    }
    try {
      const info = fstatSync(fd);
      if (!info.isFile()) return { status: 'non_regular' };
      if (info.nlink !== 1) return { status: 'hardlink_untrusted' };
      const linked = lstatSync(path);
      if (linked.dev !== info.dev || linked.ino !== info.ino || !inside(root, realpathSync(path))) {
        return { status: 'target_changed_before_read' };
      }
      if (info.size > MAX_MEMORY_BYTES) return { status: 'oversize', bytes: info.size };
      const data = Buffer.alloc(info.size + 1);
      let size = 0;
      while (size < data.length) {
        const received = readSync(fd, data, size, data.length - size);
        if (!received) break;
        size += received;
      }
      const after = fstatSync(fd);
      if (size !== info.size || after.mtimeMs !== info.mtimeMs || after.ctimeMs !== info.ctimeMs) {
        return { status: 'changed_during_read' };
      }
      return contentObject(capture, directory, data.subarray(0, size), 'file_bytes');
    } finally {
      closeSync(fd);
    }
  });
  return outcome ?? { status: 'capture_error' };
}

function hasSkill(options, section) {
  return options.messages?.some(message => message.role === 'system' &&
    message.content.some(block => block.type === 'text' && block.text.includes(section))) === true;
}

function withRequestSkill(options, anchor, section) {
  const messages = options.messages.slice();
  for (let index = messages.length - 1; index >= 0; --index) {
    const message = messages[index];
    if (message.role !== 'system') continue;
    const blockIndex = message.content.findIndex(block => block.type === 'text' && block.text.includes(anchor));
    if (blockIndex === -1) continue;
    const content = message.content.slice();
    const block = content[blockIndex];
    const end = block.text.indexOf(anchor) + anchor.length;
    content[blockIndex] = Object.freeze({
      ...block, text: `${block.text.slice(0, end)}\n\n${section}${block.text.slice(end)}`,
    });
    messages[index] = Object.freeze({ ...message, content: Object.freeze(content) });
    return Object.freeze({ ...options, messages: Object.freeze(messages) });
  }
  throw new Error('Gated Skill could not locate the benchmark system prompt');
}

// The SDK freezes request options after projecting persistent history. Decorate
// the bound stream so the normal recorder and adapter see the request-local copy.
export function installRequestSkill(ctx, anchor, section, activeQuery, marks, project) {
  const llm = ctx.llm;
  const original = llm.prepareCall;
  const prepareCall = async function (...args) {
    const ready = await original.apply(this, args);
    return Object.freeze({ ...ready, stream(options) {
      const query = options.purpose === undefined && activeQuery(options.sessionId);
      if (!query) return ready.stream(options);
      const main = marks.isAgentLoopRequest(options);
      let request = section && query.skill_active === true ? withRequestSkill(options, anchor, section) : options;
      if (main && project) request = project(request, query);
      return ready.stream(main && request !== options ? marks.markAgentLoopRequest(request) : request);
    } });
  };
  return ctx.effect(() => {
    const own = Object.getOwnPropertyDescriptor(llm, 'prepareCall');
    llm.prepareCall = prepareCall;
    return () => {
      if (Object.getOwnPropertyDescriptor(llm, 'prepareCall')?.value !== prepareCall) return;
      if (own) Object.defineProperty(llm, 'prepareCall', own);
      else delete llm.prepareCall;
    };
  }, 'clbench-submit:request-skill');
}

// DSH owns the agent loop, session history, compaction, and tool execution.
export function apply(ctx, config) {
  const capture = recorder(config);
  const attempts = new Map();
  const positions = new Map();
  const activeMemory = new Map();
  const memoryCalls = new Map();
  const states = new Map();
  const skillReceipts = new Set();
  const gated = config.skillMode === 'gated';
  const editing = config.contextEdit === true;
  const inspecting = editing || config.contextRead === true;
  const finding = inspecting || config.contextFind === true;
  const requestViews = new WeakMap();
  let skillSection;
  if (gated && (!config.skillPath || typeof config.skillAnchor !== 'string' || !config.skillAnchor)) {
    throw new Error('Gated Skill requires approved bytes and a system prompt anchor');
  }
  if (config.skillPath) {
    const bytes = readFileSync(config.skillPath);
    if (createHash('sha256').update(bytes).digest('hex') !== config.skillSha256) {
      throw new Error('Frozen skill differs from its approved hash');
    }
    skillSection = `Approved comparison skill:\n${bytes.toString('utf8')}`;
    if (!gated) ctx.systemPrompt.section({
      name: 'skill-rsi:comparison',
      order: ctx.systemPrompt.getSectionOrder('DEPLOYMENT_PERSONA_SUFFIX') + 1,
      text: skillSection,
      interpolate: false,
    });
  }
  const memoryRoot = config.captureDirectory
    ? capture.run('memory_root', () => physical(config.memoryRoot))
    : undefined;
  const activeQuery = sessionId => capture.run('active_query', () => {
    if (!existsSync(config.controlFile)) return null;
    const query = JSON.parse(readFileSync(config.controlFile, 'utf8'));
    return query.active === true && query.sessionId === sessionId && typeof query.requestId === 'string' && query.requestId
      ? query : null;
  }) ?? null;
  const queryIdentity = sessionId => activeQuery(sessionId)?.requestId ?? null;
  if (gated || inspecting) {
    // Resolve through the SDK entry point so both sides share the same request brand.
    const requireFromDsh = createRequire(realpathSync(process.argv[1]));
    installRequestSkill(ctx, config.skillAnchor, gated ? skillSection : null, activeQuery, requireFromDsh('@deepseek-ai/dsh-llm'),
      inspecting ? (options, query) => {
        const state = states.get(options.sessionId);
        if (!state?.edit || state.requestId !== query.requestId) return options;
        const view = projectContextView(options, query, state.findSources, state.edit.folded);
        requestViews.set(view.request, { ...view, version: state.edit.version });
        return view.request;
      } : null);
  }
  const refresh = agent => {
    if (!existsSync(config.controlFile)) return;
    const query = JSON.parse(readFileSync(config.controlFile, 'utf8'));
    if (query.sessionId !== agent.session.id || agent.session.header?.origin === 'subagent') return;
    const previous = states.get(agent.session.id);
    const sameQuery = previous?.requestId === query.requestId;
    if (sameQuery && previous.resultFile === query.resultFile) return;
    previous?.dispose();
    const state = { requestId: query.requestId, resultFile: query.resultFile,
      submitted: false, steps: sameQuery ? previous.steps : 0,
      restrictions: previous?.restrictions ?? [] };
    if (finding) {
      state.findSources = previous?.findSources ?? new Map();
      state.findCalls = new Map();
    }
    if (inspecting) {
      state.edit = previous?.edit ?? { version: 0, folded: new Map() };
      state.editCalls = new Map();
    }
    if (config.toolsMode === 'memory' && !previous) {
      const allowed = ctx.tools.definitions(agent).filter(tool =>
        MEMORY_TOOLS.has(tool.name)
        && tool.parameters?.properties?.file_path?.type === 'string',
      ).map(tool => tool.name);
      const root = physical(config.memoryRoot);
      state.restrictions.push(agent.ctx.tools.restrict({ allow: allowed }));
      state.restrictions.push(agent.ctx.tools.guard(exec => {
        if (exec.name === 'clbench_submit' || (finding && exec.name === 'ctx_find') ||
            (inspecting && isContextEditTool(exec.name, editing))) return;
        if (!allowed.includes(exec.name)) return 'Use clbench_submit for benchmark environment actions';
        try {
          const path = physical(resolve(dirname(root), exec.arguments.file_path));
          if (!inside(root, path)) {
            return 'Native file operations must stay within workspace/memory';
          }
        } catch (error) { return String(error?.message ?? error); }
      }));
    }
    const disposeSubmit = agent.ctx.tools.register({
      name: 'clbench_submit',
      description: 'Submit exactly one action for the current benchmark query and finish this turn. Await the environment feedback before taking another action.',
      parameters: query.schema,
      output: {
        schema: { type: 'boolean' },
        render: () => [{ type: 'text', text: 'Action submitted. Await the environment feedback.' }],
      },
      async execute(args, exec) {
        if (state.submitted) throw new Error('An action has already been submitted for this query');
        exec.signal.throwIfAborted();
        writeFileSync(query.resultFile, JSON.stringify({
          requestId: query.requestId, sessionId: agent.session.id,
          toolCallId: exec.callId, action: args,
        }), { flag: 'wx', mode: 0o600 });
        state.submitted = true;
        exec.concludeTurn();
        return true;
      },
    });
    const disposeFind = state.findSources && registerContextFind(agent, state, capture, config.maxSteps);
    const disposeEdits = inspecting && registerContextEdits(agent, state, capture, config.maxSteps, editing);
    state.dispose = () => { if (disposeEdits) disposeEdits(); disposeFind?.(); disposeSubmit(); };
    states.set(agent.session.id, state);
  };
  ctx.on('agent/created', ({ agent }) => refresh(agent));
  ctx.on('agent/inbox/inserted', ({ agent }) => refresh(agent));
  ctx.on('agent/pre-step', async ({ agent }, next) => {
    const state = states.get(agent.session.id);
    if (state && !state.submitted && ++state.steps > config.maxSteps) {
      throw new Error(`No benchmark action submitted within ${config.maxSteps} native steps`);
    }
    return next();
  });
  ctx.on('agent/disposed', ({ agent }) => {
    const state = states.get(agent.session.id);
    state?.dispose();
    for (const dispose of state?.restrictions ?? []) dispose();
    states.delete(agent.session.id);
    attempts.delete(agent.session.id);
    positions.delete(agent.session.id);
  });
  ctx.on('session/event', (session, event) => {
    const { turn, step } = event.data ?? {};
    if (event.type === 'turn/start' || event.type === 'step/start') positions.set(session.id, { turn, step });
    if (event.type === 'step/end') positions.set(session.id, { turn });
    if (event.type === 'turn/end') positions.delete(session.id);
  });
  ctx.on('agent/assistant-stream', ({ agent, frame }) => {
    const sessionId = agent.session.id;
    if (frame.type === 'start') {
      attempts.set(sessionId, { id: frame.attemptId, turn: frame.turn, step: frame.step });
    } else if (frame.type === 'end') {
      const active = attempts.get(sessionId);
      if (active?.id !== frame.attemptId) return;
      attempts.delete(sessionId);
      if (active.call) capture.event('calls', 'model_output', sessionId, {
        ...active.call, native_event_type: frame.outcome?.eventType ?? null,
        native_event_seq: frame.outcome?.seq ?? null, native_status: frame.outcome?.kind ?? null,
      });
    }
  });
  ctx.on('tools/execute', async (exec, next) => {
    if (!config.captureDirectory || !MEMORY_TOOLS.has(exec.name) || !exec.agent) return next();
    const sessionId = exec.agent.session.id;
    const operation = capture.run('memory_target', () => {
      if (!memoryRoot || physical(config.memoryRoot) !== memoryRoot) {
        throw new Error('Memory scope moved');
      }
      const root = memoryRoot;
      const requested = resolve(dirname(root), exec.arguments.file_path);
      const path = physical(requested);
      if (path === root || !inside(root, path)) return undefined;
      const peers = activeMemory.get(path) ?? new Set();
      const row = {
        operation_id: randomUUID(), query_request_id: queryIdentity(sessionId),
        tool_call_id: exec.callId, tool_name: exec.name, scope_path: relative(root, path),
        arguments: exec.arguments, observation_started_at: new Date().toISOString(),
      };
      const entry = { root, requested, path, row, peers, concurrent: peers.size > 0 };
      for (const peer of peers) peer.concurrent = true;
      peers.add(entry);
      activeMemory.set(path, peers);
      memoryCalls.set(exec.token, entry);
      capture.event('memory_operations', 'memory_before', sessionId, {
        ...row, before: snapshot(capture, config.captureDirectory, root, requested),
        proposed: exec.name === 'read' ? null : capture.run('memory_proposed', () => contentObject(
          capture, config.captureDirectory, Buffer.from(JSON.stringify(exec.arguments)), 'native_tool_arguments',
        )) ?? { status: 'capture_error' },
      });
      return entry;
    });
    try {
      return await next();
    } finally {
      if (operation) capture.run('memory_after', () => {
        const { root, requested, path, row, peers } = operation;
        const after = snapshot(capture, config.captureDirectory, root, requested);
        peers.delete(operation);
        if (!peers.size) activeMemory.delete(path);
        capture.event('memory_operations', 'memory_after', sessionId, {
          ...row, after, concurrent: operation.concurrent,
          observation_ended_at: new Date().toISOString(),
        });
      });
    }
  });
  ctx.on('tools/result', (exec, result) => {
    if (!config.captureDirectory || !MEMORY_TOOLS.has(exec.name) || !exec.agent) return;
    const sessionId = exec.agent.session.id;
    const operation = memoryCalls.get(exec.token);
    if (operation) {
      memoryCalls.delete(exec.token);
      capture.event('memory_operations', 'memory_result', sessionId, {
        ...operation.row, status: result.isError ? 'failed' : 'completed',
        canonical_value: result.value ?? null, rendered_content: result.content, error: result.error ?? null,
        readback: exec.name === 'read' && !result.isError ? capture.run('memory_readback', () => contentObject(
          capture, config.captureDirectory, Buffer.from(JSON.stringify(result.value)), 'native_read_result',
        )) ?? { status: 'capture_error' } : null,
      });
    } else {
      const path = capture.run('memory_path', () => exec.arguments?.file_path) ?? null;
      capture.event('memory_operations', 'memory_rejected', sessionId, {
        operation_id: randomUUID(), query_request_id: queryIdentity(sessionId),
        tool_call_id: exec.callId, tool_name: exec.name, requested_path: path,
        status: result.isError ? 'rejected' : 'outside_observed_scope', error: result.error ?? null,
      });
    }
  });
  // Include auxiliary calls such as compaction in the same usage ledger.
  ctx.on('llm/stream', async function* (options, next) {
    const row = {
      callId: randomUUID(), sessionId: options.sessionId,
      provider: options.provider, model: options.model, purpose: options.purpose,
      startedAt: new Date().toISOString(),
    };
    const active = attempts.get(options.sessionId);
    const nativeAttempt = options.purpose === undefined && active && !active.call ? active : undefined;
    const position = nativeAttempt ?? positions.get(options.sessionId);
    const requestPath = `request-${row.callId}.json`;
    const responsePath = `capture/responses/${row.callId}.jsonl`;
    const query = activeQuery(row.sessionId);
    const currentState = states.get(row.sessionId);
    const findState = nativeAttempt && currentState?.findSources && currentState.requestId === query?.requestId
      ? currentState : undefined;
    const view = findState && requestViews.get(options);
    const findSources = view?.sources ?? (findState && visibleSources(options.messages, query, findState.findSources));
    const exposed = gated && hasSkill(options, skillSection);
    const gateOn = gated && row.purpose === undefined && query?.skill_active === true;
    if (gated && exposed !== gateOn) throw new Error('Native Skill exposure differs from the current query gate');
    const fields = {
      llm_call_id: row.callId, query_request_id: query?.requestId ?? null,
      turn_id: position?.turn ?? null, step_id: position?.step ?? null,
      position_linkage: nativeAttempt ? 'native_assistant_attempt' : position ? 'session_scope' : null,
      native_attempt_id: nativeAttempt?.id ?? null, request_path: requestPath,
      response_path: config.captureDirectory ? responsePath : null, purpose: row.purpose ?? null,
      provider: row.provider, model: row.model,
      ...(view && { context_version: view.version, folded_sources: view.catalog.filter(item => item.folded).map(item => item.source),
        focused_sources: view.catalog.filter(item => item.focused_range).map(item => ({ source: item.source, ...item.focused_range })) }),
      ...(gated && { skill_mode: 'gated', skill_active: gateOn, skill_exposed: exposed }),
    };
    if (nativeAttempt) nativeAttempt.call = fields;
    capture.activeCalls.add(row.callId);
    capture.event('calls', 'model_start', row.sessionId, { ...fields, status: 'started' });
    const request = Object.fromEntries([
      'provider', 'model', 'system', 'systemPrompt', 'messages', 'tools', 'toolHistory',
      'temperature', 'maxTokens', 'reasoningEffort', 'stop', 'purpose', 'sessionId',
    ].filter(key => options[key] !== undefined).map(key => [key, options[key]]));
    const runtime = dirname(config.usageFile);
    const recorded = capture.run('request', () => {
      capture.save(join(runtime, requestPath), JSON.stringify(request));
      return true;
    });
    if (recorded && gated && nativeAttempt) {
      capture.event('calls', 'skill_gate', row.sessionId, {
        ...fields, sha256: config.skillSha256, method: 'request_system_prompt_section', status: 'observed_native_input',
      });
    } else if (recorded && skillSection && !skillReceipts.has(row.sessionId) && nativeAttempt &&
      capture.run('skill_receipt', () => hasSkill(request, skillSection))) {
      skillReceipts.add(row.sessionId);
      capture.event('calls', 'skill_loaded', row.sessionId, {
        ...fields, sha256: config.skillSha256, method: 'system_prompt_section', status: 'observed_native_input',
      });
    }
    const writer = { chunks: [], bytes: 0, count: 0, failed: false };
    const flush = () => {
      if (!config.captureDirectory || !writer.chunks.length || writer.failed) return;
      writer.failed = capture.run('response', () => {
        appendFileSync(join(runtime, responsePath), writer.chunks.join(''), { mode: 0o600 });
        return true;
      }) !== true;
      writer.chunks.length = writer.bytes = 0;
    };
    try {
      for await (const chunk of next()) {
        if (findState && chunk.type === 'block-end' && chunk.block?.type === 'tool-call') {
          if (chunk.block.name === 'ctx_find') findState.findCalls.set(chunk.block.id, { callId: row.callId,
            sources: findSources, version: view?.version });
          if (view && isContextEditTool(chunk.block.name)) {
            findState.editCalls.set(chunk.block.id, { callId: row.callId, name: chunk.block.name,
              catalog: view.catalog, sources: view.sources, version: view.version });
          }
        }
        if (chunk.type === 'usage') row.usage = chunk.usage;
        if (chunk.type === 'finish') row.finish = chunk.reason;
        if (config.captureDirectory && !writer.failed) {
          writer.failed = capture.run('response_chunk', () => {
            const line = `${JSON.stringify({ index: writer.count++, time_ms: Date.now(), chunk })}\n`;
            writer.chunks.push(line);
            writer.bytes += Buffer.byteLength(line);
            return true;
          }) !== true;
          if (writer.bytes >= 16384 || chunk.type === 'finish') flush();
        }
        yield chunk;
      }
    } catch (error) {
      row.error = String(error?.message ?? error);
      throw error;
    } finally {
      flush();
      capture.run('usage', () => capture.append(config.usageFile, row));
      const status = row.error || row.finish?.kind === 'error' ? 'failed'
        : row.finish?.kind === 'aborted' ? 'cancelled'
          : row.finish ? 'completed' : 'unterminated';
      capture.event('calls', 'model_end', row.sessionId, {
        ...fields, status, finish: row.finish ?? null, usage: row.usage ?? null,
        error: row.error ?? null, response_chunks: writer.count, response_capture_failed: writer.failed,
      });
      capture.activeCalls.delete(row.callId);
    }
  });
  ctx.effect(() => () => {
    try {
      for (const state of states.values()) {
        state.dispose();
        for (const dispose of state.restrictions) dispose();
      }
      states.clear();
    } finally {
      capture.health(true);
    }
  }, 'clbench-submit');
}
