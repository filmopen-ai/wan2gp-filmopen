# 1-29W1 step 1 Astra critic: API 8/10, code 8/10; logging included.
"""FilmOpen's six plain Gradio events; WanGP owns the queue and downloads."""
import json
import os
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


def log(event, **values):
    print('[filmopen] ' + event + ' ' + json.dumps(values, default=str), flush=True)


class FilmOpenPlugin(WAN2GPPlugin):
    def setup_ui(self):
        for name in ('WanGP_version', 'server_config', 'get_model_filename', 'get_model_def',
                     'get_base_model_type', 'get_model_recursive_prop', 'model_types_handlers',
                     'transformer_dtype_policy', 'download_models'):
            self.request_global(name)
        self.request_component('state')
        self.add_tab(tab_id='filmopen', label='FilmOpen', component_constructor=self.create_ui)
        self._jobs, self._session, self._download = {}, None, None
        self._lock = threading.Lock()
        log('setup.done', helper=VERSION)

    def _info(self, payload):
        import torch
        props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        cfg = self.server_config
        return dict(helper=VERSION, wangp=self.WanGP_version, gpu=props.name if props else None,
                    vramBytes=props.total_memory if props else None, profile=cfg.get('profile'),
                    quantizations=dict(transformer=cfg.get('transformer_quantization'), textEncoder=cfg.get('text_encoder_quantization')),
                    deepy=cfg.get('deepy_enabled'))

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
