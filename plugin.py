# 1-29W1 step 1 Astra critic: API 8/10, code 8/10; logging included.
"""FilmOpen's six plain Gradio events and its text endpoint; WanGP owns the queue, the downloads and the LLM."""
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from shared.utils import files_locator as fl
from shared.utils.download import _download_relpath
from shared.utils.download_progress import download_operation, DownloadCancelled

VERSION = '1.0.0'
# The FilmOpen catalogue entry of each WanGP engine the text endpoint serves (its access row's model_id).
ENTRIES = {'qwen38_27b': 'llm-qwen38-vl-27b', 'qwen38_9b': 'llm-qwen38-vl-9b'}


def log(event, **values):
    print('[filmopen] ' + event + ' ' + json.dumps(values, default=str), flush=True)


class FilmOpenPlugin(WAN2GPPlugin):
    def setup_ui(self):
        for name in ('WanGP_version', 'server_config', 'get_model_filename', 'get_model_def',
                     'get_base_model_type', 'get_model_recursive_prop', 'model_types_handlers',
                     'transformer_dtype_policy', 'download_models',
                     'ensure_prompt_enhancer_loaded', 'is_generation_in_progress', 'reset_prompt_enhancer'):
            self.request_global(name)
        self.request_component('state')
        self.add_tab(tab_id='filmopen', label='FilmOpen', component_constructor=self.create_ui)
        # WanGP's server serves it at /filmopen: a POST with `model` and `max_tokens` at the top, as FilmOpen's policy binds them.
        self.register_web_app('/filmopen', self._web_app)
        self._jobs, self._session, self._download = {}, None, None
        self._lock, self._llm = threading.Lock(), threading.Lock()
        log('setup.done', helper=VERSION)

    def _info(self, payload):
        import psutil
        import torch
        gpus = []
        for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            props, (free, _) = torch.cuda.get_device_properties(i), torch.cuda.mem_get_info(i)
            gpus.append(dict(name=props.name, vramBytes=props.total_memory, freeBytes=free))
        cfg = self.server_config
        engine = self._engine()
        # wgp.py runs as __main__ and holds the loaded LLM in this global; a plug-in's copy of it is taken before it loads.
        loaded = getattr(sys.modules.get('__main__'), 'prompt_enhancer_llm_model', None) is not None
        return dict(helper=VERSION, wangp=self.WanGP_version, gpu=gpus[0]['name'] if gpus else None,
                    vramBytes=gpus[0]['vramBytes'] if gpus else None, gpus=gpus, torch=torch.__version__,
                    cuda=torch.version.cuda, ramBytes=psutil.virtual_memory().total, profile=cfg.get('profile'),
                    quantizations=dict(transformer=cfg.get('transformer_quantization'), textEncoder=cfg.get('text_encoder_quantization')),
                    deepy=cfg.get('deepy_enabled'), llm=dict(engine=engine, entry=ENTRIES.get(engine),
                                                             quantization=cfg.get('prompt_enhancer_quantization'), loaded=loaded))

    @staticmethod
    def _quantization(engine, current):
        """The quantization WanGP's Configuration gives the two catalogue engines it switches to: for the 27B its recommendation
        for this GPU (`prompt_enhancer_quantization_ui_state`), for the 9B the current one where it takes it, else `gguf`; any
        other engine keeps the current value, and a load that fails on it is refused naming the engine."""
        if engine == 'qwen38_27b':
            from shared.deepy.onboarding import qwen38_recommended_quantization
            return qwen38_recommended_quantization() or 'gguf'
        if engine == 'qwen38_9b':
            return current if current in ('gguf', 'gguf_q8') else 'gguf'
        return current

    def _engine(self):
        """The local LLM WanGP's loader runs: `enhancer_enabled`, the local engine's id, whatever Deepy's own choice is."""
        from shared.remote_llm.config import ENHANCER_ID_TO_LOCAL_ENGINE
        return ENHANCER_ID_TO_LOCAL_ENGINE.get(self.server_config.get('enhancer_enabled'))

    def _spec(self, model):
        kind = model['modelType']
        if kind not in ('z_image', 'ti2v_2_2') or any(model[k] not in ('int8', 'bf16') for k in ('transformerQuantization', 'textEncoderQuantization')):
            raise ValueError('Unsupported model or quantization')
        definition = self.get_model_def(kind)
        weight = self.get_model_filename(kind, quantization=model['transformerQuantization'], dtype_policy=self.transformer_dtype_policy)
        urls = self.get_model_recursive_prop(kind, 'text_encoder_URLs', return_list=True, model_def=definition)
        encoder = self.get_model_filename(kind, quantization=model['textEncoderQuantization'], dtype_policy=self.transformer_dtype_policy, URLs=urls)
        return kind, definition, weight, encoder

    def _inventory(self, model):
        kind, definition, weight, encoder = self._spec(model)
        folder = definition.get('text_encoder_folder')
        files = []
        def add(path, local):
            item = dict(path=local or path, bytes=os.path.getsize(local) if local else None)
            files.append(item)
            log('models.file', modelType=kind, **item, reason='present on disk; content not checked' if local else 'not found by WanGP locator')
        for url, extra in ((weight, None), (encoder, folder)):
            path = os.path.join(extra or '', os.path.basename(url))
            add(path, fl.get_local_model_filename(url, extra_paths=extra))
        # These two selected base definitions have no LoRAs; retain their secondary URLs too.
        for prop in ('preload_URLs', 'VAE_URLs'):
            urls = self.get_model_recursive_prop(kind, prop, return_list=True, model_def=definition) if prop == 'preload_URLs' else definition.get(prop, [])
            for url in ([urls] if isinstance(urls, str) else urls or []):
                add(os.path.basename(url), fl.get_local_model_filename(url))
        handler = self.model_types_handlers[self.get_base_model_type(kind)]
        definitions = handler.query_model_files(lambda name: [os.path.basename(name)] if name else [], self.get_base_model_type(kind), definition)
        for entry in definitions if isinstance(definitions, list) else [definitions]:
            targets = entry.get('targetFolderList') or [None] * len(entry['sourceFolderList'])
            for source, names, target in zip(entry['sourceFolderList'], entry['fileList'], targets):
                for name in names:
                    path = _download_relpath(source, name, target)
                    add(path, fl.locate_file(path, error_if_none=False))
        present = sum(f['bytes'] is not None for f in files)
        return dict(modelType=kind, status='present' if present == len(files) else 'partial' if present else 'missing', files=files)

    def _models(self, payload):
        return {'models': [self._inventory(m) for m in self._payload_models(payload)]}

    def _payload_models(self, payload):
        models = payload['models']
        if not isinstance(models, list) or not 1 <= len(models) <= 2:
            raise ValueError('Supply one or two models')
        for model in models:
            self._spec(model)
        return models

    def _download_models(self, payload):
        models = self._payload_models(payload)
        with self._lock:
            if self._download and self._jobs[self._download]['state'] in ('queued', 'running'):
                log('download.busy', job=self._download, reason='one download worker at a time')
                return {'busy': self._download}
            job_id = str(uuid.uuid4())
            entry = dict(kind='download', state='queued', gen={}, progress={}, files=[], error=None, started=time.monotonic())
            self._jobs[job_id], self._download = entry, job_id
            def progress(value):
                entry['progress']['download'] = value
                log('download.progress', job=job_id, progress=value, elapsedMs=int((time.monotonic() - entry['started']) * 1000))
            entry['gen']['download_progress_callback'] = progress
            def work():
                entry['state'] = 'running'
                log('download.running', job=job_id, models=models)
                try:
                    with download_operation(entry['gen']):
                        for model in models:
                            inventory = self._inventory(model)
                            if inventory['status'] == 'present':
                                log('download.skip', job=job_id, inventory=inventory, reason='all files present')
                                continue
                            kind, definition, weight, encoder = self._spec(model)
                            log('download.model', job=job_id, model=model, weight=weight, encoder=encoder, reason=inventory['status'])
                            self.download_models(weight, kind, model_def=definition, gen=entry['gen'])
                            self.download_models(encoder, kind, 2, -1, force_path=definition.get('text_encoder_folder'), model_def=definition, gen=entry['gen'])
                    entry['state'] = 'done'
                except DownloadCancelled as error:
                    entry.update(state='cancelled', error=str(error))
                except Exception as error:
                    entry.update(state='failed', error=f'{type(error).__name__}: {error}')
                    log('download.fail', job=job_id, error=repr(error), trace=traceback.format_exc())
                log('download.end', job=job_id, state=entry['state'], error=entry['error'], elapsedMs=int((time.monotonic() - entry['started']) * 1000))
            entry['thread'] = threading.Thread(target=work, name='FilmOpen download', daemon=True)
            entry['thread'].start()
        return {'download': job_id}

    def _generate(self, payload):
        client_id = str(payload['clientId'])
        settings = dict(payload['settings'], client_id=client_id)
        entry = dict(kind='generate', progress={}, started=time.monotonic())
        class Callbacks:
            def on_status(self, status):
                entry['progress']['status'] = str(status)
                log('job.status', job=client_id, status=str(status))
            def on_progress(self, update):
                entry['progress'].update(phase=update.phase, step=update.current_step, steps=update.total_steps, status=update.status)
                log('job.progress', job=client_id, progress=entry['progress'])
            def on_error(self, error):
                log('job.error', job=client_id, error=str(error), stage=getattr(error, 'stage', None))
            def on_complete(self, result):
                log('job.complete', job=client_id, success=result.success, cancelled=result.cancelled, files=result.generated_files, errors=[str(e) for e in result.errors], elapsedMs=int((time.monotonic() - entry['started']) * 1000))
        with self._lock:
            if self._session is None:
                from shared.api import WanGPSession
                service = self.state.value.service
                log('session.create', reason='WanGP in-process worker', settings=settings)
                self._session = WanGPSession(webui_state=service._state, console_output=False, console_isatty=False, webui_append=True)
                self._session._gradio_webui_context = {'load_queue': service.load_inline_queue}
            entry['job'] = self._session.submit_task(settings, callbacks=Callbacks())
            self._jobs[client_id] = entry
        return {'job': client_id}

    def _job(self, payload):
        job_id = str(payload['job'])
        entry = self._jobs.get(job_id)
        if entry is None:
            return dict(kind=None, state='unknown', progress=None, files=[], error=None)
        if entry['kind'] == 'download':
            return {k: entry[k] for k in ('kind', 'state', 'progress', 'files', 'error')}
        job, gen = entry['job'], self.state.value.service._state['gen']
        if job.done:
            result = job.result()
            state = 'done' if result.success and result.generated_files else 'cancelled' if result.cancelled else 'failed'
            return dict(kind='generate', state=state, progress=entry['progress'], files=result.generated_files, error=[str(e) for e in result.errors] or None)
        queue = [t.get('params', {}).get('client_id') for t in gen.get('queue', []) or [] if isinstance(t, dict)]
        running = bool(queue) and queue[0] == job_id and bool(gen.get('in_progress'))
        return dict(kind='generate', state='running' if running else 'queued', progress=dict(entry['progress'], download=gen.get('download_progress') if running else None), files=[], error=None)

    def _cancel(self, payload):
        job_id = str(payload['job'])
        entry = self._jobs.get(job_id)
        if entry is None:
            return {'state': 'unknown'}
        if self._job(payload)['state'] in ('done', 'failed', 'cancelled'):
            return {'state': 'finished'}
        if entry['kind'] == 'download':
            entry['gen']['abort'] = True
        else:
            entry['job'].cancel()
        log('job.cancel', job=job_id, reason='caller requested cancellation; terminal observation follows')
        return {'state': 'requested'}

    def _web_app(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        app = FastAPI()
        @app.post('/v1/chat/completions')
        def chat(body: dict):  # A plain def: FastAPI runs it on a worker thread, so a long generation blocks no other request.
            status, answer = self._complete(body)
            return JSONResponse(answer, status_code=status)
        return app

    def _complete(self, body):
        """One completion on the engine WanGP is set to: {text, tool_calls, usage, finish}, or {error, message}."""
        started, stage = time.monotonic(), 'engine'
        def answer(status, value):
            log('complete.answer', status=status, answer=value, elapsedMs=int((time.monotonic() - started) * 1000))
            return status, value
        log('complete.call', body=body)
        try:
            from shared.remote_llm.config import LOCAL_ENGINE_TO_ENHANCER_ID
            requested, switch = body.get('model'), False
            wanted = LOCAL_ENGINE_TO_ENHANCER_ID.get(requested)
            if wanted is None:
                log('complete.engine', requested=requested, decision='unknown')
                return answer(409, dict(error='unknown-engine', message=f'WanGP has no local LLM named {requested}'))
            if self.is_generation_in_progress() or not self._llm.acquire(blocking=False):
                log('complete.decided', decision='busy', reason='a render runs' if self.is_generation_in_progress() else 'another completion runs')
                return answer(409, dict(error='busy', message='WanGP is busy rendering or answering; try again when it finishes'))
            cfg = self.server_config
            try:
                stage = 'input'
                tools, maximum = body.get('tools') or None, int(body['max_tokens'])
                messages = [self._message(m) for m in body['messages']]
                stage = 'load'
                log('complete.load.start', engine=requested)
                # Decision 1, item 4: the engine the request names. WanGP's loader reads it from WanGP's shared config, so WanGP's
                # engine, the values to put back and the switch are read and decided under WanGP's own config lock, and the
                # requested values stand there only while it loads (review 1, M2): a save WanGP makes meanwhile waits, and writes
                # WanGP's own values. The engine stays loaded for this answer; a reset after it makes WanGP's next own use load its
                # engine again.
                from shared.utils.config_store import config_lock
                waited = time.monotonic()
                with config_lock:
                    held = time.monotonic()
                    try:
                        actual = self._engine()
                        own = (cfg.get('enhancer_enabled', 0), cfg.get('prompt_enhancer_quantization'))
                        switch = requested != actual
                        log('complete.engine', requested=requested, actual=actual, own=own, decision='switch' if switch else 'run',
                            waitedMs=int((held - waited) * 1000), reason="read under WanGP's config lock; its saves wait while the loader runs")
                        if switch:
                            cfg['enhancer_enabled'], cfg['prompt_enhancer_quantization'] = wanted, self._quantization(requested, own[1])
                            self.reset_prompt_enhancer()
                            log('complete.engine.switched', engine=requested, enhancerEnabled=wanted, quantization=cfg['prompt_enhancer_quantization'])
                        try:
                            model, _ = self.ensure_prompt_enhancer_loaded()
                        except Exception as error:
                            raise RuntimeError(f'WanGP could not load {requested}: {type(error).__name__}: {error}') from error
                        finally:
                            if switch:
                                cfg['enhancer_enabled'], cfg['prompt_enhancer_quantization'] = own
                                log('complete.engine.restored', engine=actual, enhancerEnabled=own[0], quantization=own[1])
                    finally:
                        log('complete.engine.unlocked', engine=requested, heldMs=int((time.monotonic() - held) * 1000))
                log('complete.load.end', engine=requested, elapsedMs=int((time.monotonic() - started) * 1000))
                from shared.prompt_enhancer.qwen35_assistant_runtime import (Qwen35AssistantRuntime, extract_tool_calls,
                    render_assistant_messages, strip_tool_blocks, strip_trailing_stop_markup)
                stage = 'render'
                runtime = Qwen35AssistantRuntime(model)
                ids = render_assistant_messages(runtime.tokenizer, messages, tools, add_generation_prompt=True, thinking_enabled=False)
                log('complete.input', messages=messages, tools=tools, format='native', inputTokens=len(ids), maxTokens=maximum)
                stage = 'generate'
                # The engine is sized at priming from the model's minimum context: WanGP's loader leaves the prompt enhancer's 8000
                # (qwen35_text.py:1472), and a longer prompt then has one token to answer. As Deepy does (shared/deepy/engine.py:7891),
                # the hint is this prompt and its maximum answer; WanGP's value is put back after the answer.
                was = getattr(model, '_prompt_enhancer_min_model_len_hint', None)
                model._prompt_enhancer_min_model_len_hint = len(ids) + maximum
                log('complete.context', inputTokens=len(ids), maxTokens=maximum, hint=len(ids) + maximum, was=was)
                try:
                    runtime.prime_context(ids, seed=0)
                    result = runtime.generate_segment(max_new_tokens=maximum, seed=0, do_sample=False, temperature=None,
                                                      top_p=None, top_k=None, thinking_enabled=False)
                finally:
                    model._prompt_enhancer_min_model_len_hint = was
                log('complete.output', raw=result.raw_text, stop=result.stop_reason, outputTokens=result.token_count)
            finally:
                if switch:
                    self.reset_prompt_enhancer()  # WanGP's next own use loads its engine again.
                    log('complete.engine.reset', engine=requested, reason="WanGP's next own use loads its own engine")
                self._llm.release()
            usage = dict(inputTokens=len(ids), outputTokens=result.token_count)
            # Plan 2, M4, before any decoding (review 1, M3): an answer stopped at its maximum is refused where tools were offered,
            # where it holds tool markup (offered or not) or where no words came; only words alone keep `length`.
            if result.stop_reason in ('max_tokens', 'context_limit'):
                marked = re.search(r'<\s*/?\s*tool_call\s*>', result.raw_text, re.IGNORECASE) is not None  # as WanGP's own scanner reads a tag
                words = strip_trailing_stop_markup(result.raw_text).strip()
                if tools or marked or not words:
                    log('complete.decided', decision='cutOff', stop=result.stop_reason, tools=bool(tools), calls=marked, words=len(words))
                    return answer(422, dict(error='cutOff', message=f'The local model stopped at its output maximum ({maximum} tokens) before it finished', usage=usage))
            stage = 'decode'
            calls = extract_tool_calls(result.raw_text) if tools else []
            text = strip_trailing_stop_markup(strip_tool_blocks(result.raw_text) if calls else result.raw_text).strip()
            finish = 'tool_calls' if calls else 'length' if result.stop_reason in ('max_tokens', 'context_limit') else 'stop'
            log('complete.decided', decision='calls' if calls else 'length' if finish == 'length' else 'words', stop=result.stop_reason, calls=len(calls))
            return answer(200, dict(text=text, finish=finish, usage=usage, tool_calls=[
                dict(id=f'call_{i}', type='function', function=dict(name=c['name'], arguments=json.dumps(c['arguments'])))
                for i, c in enumerate(calls, 1)]))
        except Exception as error:
            log('complete.fail', stage=stage, error=repr(error), trace=traceback.format_exc())
            return answer(500, dict(error='failed', message=str(error) if stage == 'load' else f'{type(error).__name__}: {error}'))

    @staticmethod
    def _message(message):
        """A chat-completions message as Qwen's chat template takes it: text parts joined, call arguments as objects."""
        content = message.get('content')
        if isinstance(content, list):
            content = '\n'.join(part.get('text', '') for part in content if part.get('type') == 'text')
        rewritten = dict(role=message['role'], content=content or '')
        if message.get('tool_calls'):
            rewritten['tool_calls'] = [dict(call, function=dict(call['function'], arguments=json.loads(call['function']['arguments'])))
                                       for call in message['tool_calls']]
        if message.get('tool_call_id'):
            rewritten['tool_call_id'] = message['tool_call_id']
        return rewritten

    def create_ui(self):
        gr.Markdown('FilmOpen helper ' + VERSION)
        for name, handler in (('info', self._info), ('models', self._models), ('download', self._download_models), ('generate', self._generate), ('job', self._job), ('cancel', self._cancel)):
            box_in, box_out = gr.Textbox(visible=False), gr.Textbox(visible=False)
            def run(text, name=name, handler=handler):
                started = time.monotonic()
                log('filmopen_' + name + '.call', text=text)
                try:
                    answer = handler(json.loads(text))
                except Exception as error:
                    log('filmopen_' + name + '.fail', error=repr(error), trace=traceback.format_exc())
                    answer = {'error': f'{type(error).__name__}: {error}'}
                log('filmopen_' + name + '.answer', answer=answer, elapsedMs=int((time.monotonic() - started) * 1000))
                return json.dumps(answer, default=str)
            box_in.submit(run, inputs=[box_in], outputs=[box_out], api_name='filmopen_' + name)
