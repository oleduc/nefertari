import sys
import logging
import math
from argparse import ArgumentParser
from multiprocessing import Pool, Process, Manager
from datetime import datetime

from pyramid.paster import bootstrap
from pyramid.config import Configurator
from sqlalchemy import text
from six.moves import urllib

from nefertari.utils import dictset, split_strip, to_dicts, to_indexable_dicts
from nefertari.elasticsearch import ES, ESActionRegistry, create_index_with_settings
from nefertari import engine


def main(argv=sys.argv):

    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
            '-c', '--config', help='config.ini (required)',
            required=True)
    parser.add_argument(
            '--params', help='Url-encoded params for each model')
    parser.add_argument('--index', help='Index name', default=None)
    parser.add_argument(
            '--chunk',
            help=('Index chunk size. If chunk size not provided '
                  '`elasticsearch.chunk_size` setting is used'),
            type=int)

    parser.add_argument('--processes',
                        required=False,
                        help='Split es action by multiple process',
                        action='store',
                        default=1,
                        type=int)

    parser.add_argument('--debug',
                        required=False,
                        help='Enable debug mode to check count for processed and existed item',
                        action='store_true',
                        default=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
            '--models',
            help=('Comma-separated list of model names to index'))
    group.add_argument(
            '--recreate',
            help='Recreate index and reindex all documents',
            action='store_true',
            default=False)

    options = parser.parse_args()

    if not options.config:
        return parser.print_help()

    app_registry = setup_app(options=options, put_mappings=True)

    processes = options.processes

    if options.recreate:
        recreate_index(app_registry)
        models = engine.get_document_classes()
        model_names = [
                name for name, model in models.items()
                if getattr(model, '_index_enabled', False)]
    else:
        model_names = split_strip(options.models)

    queue = get_processing_queue(model_names, processes, options)
    pool = Pool(processes)
    processors = map(lambda i: ESProcessor(options=options, queue=queue, pid=i),
                     range(0, processes))

    if options.debug:
        _check_results(pool.map(ESProcessor.apply, processors))
    else:
        pool.map(ESProcessor.apply, processors)


def _check_results(result):
    initial_dict = {}
    flat_list = []
    log = get_logger()
    log.setLevel(logging.INFO)

    for item in result:
        flat_list.extend(item)

    for item in flat_list:
        for key in item:
            if key not in initial_dict:
                initial_dict[key] = []
            initial_dict[key].extend(item[key])

    for model_name in initial_dict:
        items_count = engine.get_document_cls(model_name).get_collection().count()
        log.info('Total items count {} for model {}'.format(items_count, model_name))
        log.info('Total indexed tems count {} for model {}'.format(len(set(initial_dict[model_name])), model_name))

        assert len(initial_dict[model_name]) == len(set(initial_dict[model_name]))
        assert len(initial_dict[model_name]) == items_count


def get_logger():
    log = logging.getLogger()
    log.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def setup_app(options, put_mappings):
    # Prevent ES.setup_mappings running on bootstrap;
    # Restore ES._mappings_setup after bootstrap is over
    log = get_logger()
    mappings_setup = getattr(ES, '_mappings_setup', False)
    try:
        ES._mappings_setup = True
        env = bootstrap(options.config)
    finally:
        ES._mappings_setup = mappings_setup
    registry = env['registry']
    # Include 'nefertari.engine' to setup specific engine
    config = Configurator(settings=registry.settings)
    config.include('nefertari.engine')
    ES.setup(dictset(registry.settings))

    if put_mappings:
        log.setLevel(logging.INFO)
        ES.setup_mappings()
    return registry

def recreate_index(registry):
    log = get_logger()
    settings = dictset(registry.settings)
    log.info('Deleting index')
    ES.delete_index()
    log.info('Creating index')
    create_index_with_settings(settings)
    log.info('Creating mappings')
    ES.setup_mappings()


def split_collection(limit, n, collection):
    list_size = (math.ceil(limit / n) - 1) or 1
    iteration = 0

    for i in range(0, limit, list_size):
        iteration += 1

        if iteration == n:
            yield collection[i:]
            break

        yield collection[i: i + list_size]


def get_processing_queue(model_names, processes_count, options):
    manager = Manager()
    queue = manager.Queue()

    def _get_chunks(model_names, processes_count, q):
        setup_app(options, put_mappings=False)
        from nefertari import engine
        from pyramid_sqlalchemy import Session
        model_chunks = {}

        for model_name in model_names:
            model = engine.get_document_cls(model_name)
            limit = model.get_collection().count()
            table_name = model.__tablename__
            statement = text('SELECT {} FROM public.{}'.format(model.pk_field(), table_name))
            query = Session().query(model).from_statement(statement)
            items = list(map(lambda item: getattr(item, model.pk_field()) ,sorted(query.values(model.pk_field()))))
            chunks = list(split_collection(limit, processes_count, items))
            model_chunks[model_name] = chunks

            for p in range(0, processes_count):
                if len(chunks) > p:
                    q.put((model_name, chunks[p]), block=False)
    p = Process(target=_get_chunks, args=(model_names, processes_count, queue), daemon=True)
    p.start()
    return queue


class ESProcessor(object):

    def __init__(self, options, queue, pid):
        self.options = options
        self.queue = queue
        self.id = pid

    def __call__(self):
        log = get_logger()
        results = []

        setup_app(self.options, put_mappings=False)

        while True:
            model_name, ids = self.queue.get()
            results.append({model_name: self.index_model(model_name, ids)})

            if self.queue.empty():
                break

        log.info('indexing finished for process {} at {}'.format(self.id, str(datetime.now())))
        return results

    def index_model(self, model_name, ids):
        from nefertari import engine
        from pyramid_sqlalchemy import Session

        model = engine.get_document_cls(model_name)

        chunk_size = int(self.options.chunk or len(ids))

        es = ES(source=model_name, index_name=self.options.index, chunk_size=chunk_size)

        query_set = Session().query(model).filter(getattr(model, model.pk_field()).in_(ids)).all()
        ids = [getattr(item, item.pk_field()) for item in query_set]
        documents = to_indexable_dicts(query_set)

        es.index_missing_documents(documents)
        es_actions = ESActionRegistry()

        for actions in es_actions.registry.values():
            es_actions.force_indexation(actions=actions)

        es_actions.registry.clear()
        return ids

    @staticmethod
    def apply(processor):
        return processor()
