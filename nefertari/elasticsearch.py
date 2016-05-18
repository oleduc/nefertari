from __future__ import absolute_import
import json
import logging
from functools import partial
from collections import defaultdict

import elasticsearch
from elasticsearch import helpers
import os
import six

from nefertari.utils import (
    dictset, dict2proxy, process_limit, split_strip, to_dicts, DataProxy)
from nefertari.json_httpexceptions import (
    JHTTPBadRequest, JHTTPNotFound, exception_response)
from nefertari import engine, RESERVED_PARAMS

log = logging.getLogger(__name__)


class IndexNotFoundException(Exception):
    pass


class ESHttpConnection(elasticsearch.Urllib3HttpConnection):
    def _catch_index_error(self, response):
        """ Catch and raise index errors which are not critical and thus
        not raised by elasticsearch-py.
        """
        code, headers, raw_data = response
        if not raw_data:
            return
        data = json.loads(raw_data)
        if not data or not data.get('errors'):
            return
        try:
            error_dict = data['items'][0]['index']
            message = error_dict['error']
        except (KeyError, IndexError):
            return
        raise exception_response(400, detail=message)

    def perform_request(self, *args, **kw):
        try:
            if log.level == logging.DEBUG:
                msg = str(args)
                if len(msg) > 512:
                    msg = msg[:300] + '...TRUNCATED...' + msg[-212:]
                log.debug(msg)
            resp = super(ESHttpConnection, self).perform_request(*args, **kw)
        except Exception as e:
            log.error(e.error)
            status_code = e.status_code
            if status_code == 404 and 'IndexMissingException' in e.error:
                raise IndexNotFoundException()
            if status_code == 'N/A':
                status_code = 400
            raise exception_response(
                status_code,
                explanation=six.b(e.error),
                extra=dict(data=e))
        else:
            self._catch_index_error(resp)
            return resp


def includeme(config):
    Settings = dictset(config.registry.settings)
    ES.setup(Settings)

    # Load custom index settings
    index_settings = None
    index_settings_path = None

    if "elasticsearch.index.settings_file" in Settings:
        index_settings_path = Settings["elasticsearch.index.settings_file"]

        if not os.path.exists(index_settings_path):
            raise Exception("Custom index settings file does not exist : '{file_name}'".format(
                file_name=index_settings_path
            ))
    else:
        if os.path.exists("index_settings.json"):
            index_settings_path = "index_settings.json"

    if index_settings_path is not None:
        with open(index_settings_path) as data_file:
            try:
                index_settings = json.load(data_file)
            except:
                raise Exception("Could not parse custom index settings : '{file_name}'".format(
                    file_name=index_settings_path
                ))

    ES.create_index(index_settings=index_settings)
    if ES.settings.asbool('enable_polymorphic_query'):
        config.include('nefertari.polymorphic')


def _bulk_body(documents_actions, request):
    kwargs = {
        'client': ES.api,
        'actions': documents_actions,
    }

    if request is None:
        query_params = {}
    else:
        query_params = request.params.mixed()

    query_params = dictset(query_params)
    refresh_enabled = ES.settings.asbool('enable_refresh_query')

    if '_refresh_index' in query_params and refresh_enabled:
        kwargs['refresh'] = query_params.asbool('_refresh_index')

    executed_num, errors = helpers.bulk(**kwargs)
    log.info('Successfully executed {} Elasticsearch action(s)'.format(
        executed_num))
    if errors:
        raise Exception('Errors happened when executing Elasticsearch '
                        'actions'.format('; '.join(errors)))


def process_fields_param(fields):
    """ Process 'fields' ES param.

    * Fields list is split if needed
    * '_type' field is added, if not present, so the actual value is
      displayed instead of 'None'
    """
    if not fields:
        return fields
    if isinstance(fields, six.string_types):
        fields = split_strip(fields)
    if '_type' not in fields:
        fields.append('_type')
    return {
        '_source_include': fields,
        '_source': True,
    }


def apply_sort(_sort):
    _sort_param = []

    if _sort:
        for each in [e.strip() for e in _sort.split(',')]:
            if each.startswith('-'):
                _sort_param.append(each[1:] + ':desc')
            elif each.startswith('+'):
                _sort_param.append(each[1:] + ':asc')
            else:
                _sort_param.append(each + ':asc')

    return ','.join(_sort_param)


def build_terms(name, values, operator='OR'):
    return (' %s ' % operator).join(['%s:%s' % (name, v) for v in values])


class _ESDocs(list):
    def __init__(self, *args, **kw):
        self._total = 0
        self._start = 0
        super(_ESDocs, self).__init__(*args, **kw)


class ES(object):
    api = None
    settings = None
    document_proxies = {}

    @classmethod
    def src2type(cls, source):
        """ Convert string :source: to ES document _type name. """
        return source

    @classmethod
    def setup(cls, settings):
        cls.settings = settings.mget('elasticsearch')
        cls.settings.setdefault('chunk_size', 500)
        cls.document_proxies = {}

        try:
            _hosts = cls.settings.hosts
            hosts = []
            for (host, port) in [
                split_strip(each, ':') for each in split_strip(_hosts)]:
                hosts.append(dict(host=host, port=port))

            params = {}
            if cls.settings.asbool('sniff'):
                params = dict(
                    sniff_on_start=True,
                    sniff_on_connection_fail=True
                )

            cls.api = elasticsearch.Elasticsearch(
                hosts=hosts, serializer=engine.ESJSONSerializer(),
                connection_class=ESHttpConnection, **params)
            log.info('Including Elasticsearch. %s' % cls.settings)

        except KeyError as e:
            raise Exception(
                'Bad or missing settings for elasticsearch. %s' % e)

    def __init__(self, source='', index_name=None, chunk_size=None):
        self.doc_type = self.src2type(source)

        if self.doc_type in ES.document_proxies:
            self.proxy = ES.document_proxies[self.doc_type]
        else:
            self.proxy = None

        self.index_name = index_name or self.settings.index_name
        if chunk_size is None:
            chunk_size = self.settings.asint('chunk_size')
        self.chunk_size = chunk_size

    @classmethod
    def create_index(cls, index_name=None, index_settings=None):
        index_name = index_name or cls.settings.index_name
        try:
            cls.api.indices.exists([index_name])
        except (IndexNotFoundException, JHTTPNotFound):
            cls.api.indices.create(
                index=index_name,
                body={
                    'settings': index_settings
                }
            )

    @classmethod
    def delete_index(cls, index_name=None):
        index_name = index_name or cls.settings.index_name
        try:
            cls.api.indices.delete([index_name])
        except (IndexNotFoundException, JHTTPNotFound):
            return

    @classmethod
    def setup_mappings(cls, force=False):
        """ Setup ES mappings for all existing models.

        This method is meant to be run once at application lauch.
        ES._mappings_setup flag is set to not run make mapping creation
        calls on subsequent runs.

        Use `force=True` to make subsequent calls perform mapping
        creation calls to ES.
        """
        if getattr(cls, '_mappings_setup', False) and not force:
            log.debug('ES mappings have been already set up for currently '
                      'running application. Call `setup_mappings` with '
                      '`force=True` to perform mappings set up again.')
            return
        log.info('Setting up ES mappings for all existing models')
        models = engine.get_document_classes()

        try:
            for model_name, model_cls in models.items():
                if getattr(model_cls, '_index_enabled', False):
                    mapping, substitutions = model_cls.get_es_mapping()
                    cls.setup_document_proxy(model_cls.__name__, substitutions)

                    es = cls(model_cls.__name__)
                    es.put_mapping(body=mapping)
        except JHTTPBadRequest as ex:
            raise Exception(ex.json['extra']['data'])

        cls._mappings_setup = True

    @classmethod
    def setup_document_proxy(cls, type_name, substitutions):
        cls.document_proxies[type_name] = type(type_name, (DataProxy,), {
            "__init__": DataProxy.__init__,
            "__setattr__": DataProxy.__setattr__,
            "substitutions": list(),
            "to_dict": DataProxy.to_dict
        })

        if len(substitutions) > 0:
            cls.document_proxies[type_name].substitutions = substitutions

    def put_mapping(self, body, **kwargs):
        self.api.indices.put_mapping(
            doc_type=self.doc_type,
            body=body,
            index=self.index_name,
            **kwargs)

    def process_chunks(self, documents, operation):
        """ Apply `operation` to chunks of `documents` of size
        `self.chunk_size`.

        """
        chunk_size = self.chunk_size
        start = end = 0
        count = len(documents)

        while count:
            if count < chunk_size:
                chunk_size = count
            end += chunk_size

            bulk = documents[start:end]
            operation(documents_actions=bulk)

            start += chunk_size
            count -= chunk_size

    def build_qs(self, params, _raw_terms='', operator='AND'):
        # if param is _all then remove it
        params.pop_by_values('_all')

        terms = []

        for k, v in params.items():
            if k.startswith('__'):
                continue

            key = k

            # Substitute nested key names
            if '.' in key:
                key_terms = key.split('.')

                if len(key_terms) >= 1 and key_terms[0] in self.proxy.substitutions:
                    key_terms[0] += "_nested"
                    key = '.'.join(key_terms)

            if type(v) is list:
                terms.append(build_terms(key, v))
            else:
                terms.append('%s:%s' % (key, v))

        terms = sorted([term for term in terms if term])
        _terms = (' %s ' % operator).join(terms)
        if _raw_terms:
            add = (' AND ' + _raw_terms) if _terms else _raw_terms
            _terms += add
        return _terms

    def prep_bulk_documents(self, action, documents):
        if not isinstance(documents, list):
            documents = [documents]

        docs_actions = []
        for doc in documents:
            if not isinstance(doc, dict):
                raise ValueError(
                    'Document type must be `dict` not a `{}`'.format(
                        type(doc).__name__))

            if '_type' in doc:
                _doc_type = self.src2type(doc.pop('_type'))
            else:
                _doc_type = self.doc_type

            doc_action = {
                '_op_type': action,
                '_index': self.index_name,
                '_type': _doc_type,
                '_id': doc['_pk'],
                '_source': doc,
            }

            docs_actions.append(doc_action)

        return docs_actions

    def _bulk(self, action, documents, request=None):
        if not documents:
            log.debug('Empty documents: %s' % self.doc_type)
            return

        documents_actions = self.prep_bulk_documents(action, documents)

        if documents_actions:
            operation = partial(_bulk_body, request=request)
            self.process_chunks(
                documents=documents_actions,
                operation=operation)
        else:
            log.warning('Empty body')

    def index(self, documents, request=None, **kwargs):
        if isinstance(documents, list):
            self.index_documents(documents, request=request)
        elif isinstance(documents, set):
            self.index_documents(list(documents), request=request)
        elif engine.is_object_document(documents):
            self._bulk('index', documents.to_indexable_dict(), request)
        else:
            raise TypeError(
                'Documents type must be `list`,`set` or `BaseDocument` not a `{}`'.format(
                    type(documents).__name__))

    def index_document(self, document, request=None):
        if engine.is_object_document(document):
            """ Reindex all `document`s. """
            self._bulk('index', document.to_indexable_dict(), request)
        else:
            raise TypeError(
                'Document type must be an instance of a type extending `BaseDocument` not a `{}`'.format(
                    type(document).__name__))

    def index_documents(self, documents, request=None):
        dict_documents = []

        for document in documents:
            if engine.is_object_document(document):
                dict_documents.append(document.to_indexable_dict())
            else:
                raise TypeError("nefertari.elasticsearch.index_documents takes a list of documents extending "
                                "BaseDocument")

        """ Reindex all `document`s. """
        self._bulk('index', dict_documents, request)

    def index_nested_document(self, parent, field, target, request=None):
        actions = []
        _doc_type = self.src2type(getattr(parent, '_type', self.doc_type))
        target_field = getattr(parent, field)
        _field_name = field + "_nested" if field in parent._nested_relationships else field

        if isinstance(target_field, list):
            action = {
                '_op_type': 'update',
                '_index': self.index_name,
                '_type': _doc_type,
                '_id': parent.id,
                'script': {
                    "file": "nested_update",
                    "params": {
                        "field_name": _field_name,
                        "nested_document": target.to_dict(_depth=0)
                    }
                }
            }
            actions.append(action)
        else:
            raise Exception("A nested document that is not in a list should not use partial update.")

        _bulk_body(actions, request)

    def index_missing_documents(self, documents, request=None):
        """ Index documents that are missing from ES index.

        Determines which documents are missing using ES `mget` call which
        returns a list of document IDs as `documents`. Then missing
        `documents` from that list are indexed.
        """
        log.info('Trying to index documents of type `{}` missing from '
                 '`{}` index'.format(self.doc_type, self.index_name))

        if not documents:
            log.info('No documents to index')
            return
        query_kwargs = dict(
            index=self.index_name,
            doc_type=self.doc_type,
            fields=['_id'],
            body={'ids': [d['_pk'] for d in documents]},
        )
        try:
            response = self.api.mget(**query_kwargs)
        except IndexNotFoundException:
            indexed_ids = set()
        else:
            indexed_ids = set(
                d['_id'] for d in response['docs'] if d.get('found'))
        documents = [d for d in documents if str(d['_pk']) not in indexed_ids]

        if not documents:
            log.info('No documents of type `{}` are missing from '
                     'index `{}`'.format(self.doc_type, self.index_name))
            return

        self._bulk('index', documents, request)

    def delete(self, ids, request=None):
        if not isinstance(ids, list):
            ids = [ids]

        documents = [{'_pk': _id, '_type': self.doc_type} for _id in ids]
        self._bulk('delete', documents, request=request)

    def get_by_ids(self, ids, **params):
        if not ids:
            return _ESDocs()

        _raise_on_empty = params.pop('_raise_on_empty', False)
        fields = params.pop('_fields', [])

        _limit = params.pop('_limit', len(ids))
        _page = params.pop('_page', None)
        _start = params.pop('_start', None)
        _start, _limit = process_limit(_start, _page, _limit)

        docs = []
        for _id in ids:
            docs.append(
                dict(
                    _index=self.index_name,
                    _type=self.src2type(_id['_type']),
                    _id=_id['_id']
                )
            )

        params = dict(
            body=dict(docs=docs)
        )
        if fields:
            fields_params = process_fields_param(fields)
            params.update(fields_params)

        documents = _ESDocs()
        documents._nefertari_meta = dict(
            start=_start,
            fields=fields,
        )

        try:
            data = self.api.mget(**params)
        except IndexNotFoundException:
            if _raise_on_empty:
                raise JHTTPNotFound(
                    '{}({}) resource not found (Index does not exist)'.format(
                        self.doc_type, params))
            documents._nefertari_meta.update(total=0)
            return documents

        for found_doc in data['docs']:
            try:
                output_doc = found_doc['_source']
                output_doc['_type'] = found_doc['_type']
            except KeyError:
                msg = "ES: '%s(%s)' resource not found" % (
                    found_doc['_type'], found_doc['_id'])
                if _raise_on_empty:
                    raise JHTTPNotFound(msg)
                else:
                    log.error(msg)
                    continue

            documents.append(dict2proxy(dictset(output_doc), ES.document_proxies[found_doc['_type']]))

        documents._nefertari_meta.update(
            total=len(documents),
        )

        return documents

    def substitute_nested_fields(self, fields, delimiter, first_only=False):
        terms = fields.split(delimiter)

        if self.proxy is not None and len(terms) > 0:
            for index in range(0, 1 if first_only else len(terms)):
                has_modifier = terms[index].startswith('-') or terms[index].startswith('+')

                if has_modifier:
                    is_substituted = terms[index][1:] in self.proxy.substitutions
                else:
                    is_substituted = terms[index] in self.proxy.substitutions

                if is_substituted:
                    terms[index] += "_nested"
                    fields = delimiter.join(terms)

        return fields

    def add_nested_fields(self, fields, delimiter):
        terms = fields
        nested_fields = []

        if self.proxy is None or len(self.proxy.substitutions) == 0:
            return fields

        if isinstance(fields, str):
            terms = fields.split(delimiter)

        for term in terms:
            if term in self.proxy.substitutions:
                nested_fields.append(term + "_nested")

        return delimiter.join(terms + nested_fields)

    def build_search_params(self, params):
        params = dictset(params)

        _params = dict(
            index=self.index_name,
            doc_type=self.doc_type
        )
        _raw_terms = params.pop('q', '')

        if 'body' not in params:
            query_string = self.build_qs(params.remove(RESERVED_PARAMS), _raw_terms)
            if query_string:
                _params['body'] = {
                    'query': {
                        'query_string': {
                            'query': query_string
                        }
                    }
                }
            else:
                _params['body'] = {"query": {"match_all": {}}}
        else:
            _params['body'] = params['body']

        if '_limit' not in params:
            params['_limit'] = self.api.count()['count']

        _params['from_'], _params['size'] = process_limit(
            params.get('_start', None),
            params.get('_page', None),
            params['_limit'])

        if '_sort' in params:
            params['_sort'] = self.substitute_nested_fields(params['_sort'], '.', first_only=True)
            _params['sort'] = apply_sort(params['_sort'])

        if '_fields' in params:
            params['_fields'] = self.add_nested_fields(params['_fields'], ',')
            _params['fields'] = params['_fields']

        if '_search_fields' in params:
            search_fields = params['_search_fields'].split(',')

            search_fields.reverse()

            # Substitute search fields and add ^index
            for index, search_field in enumerate(search_fields):
                sf_terms = search_field.split('.')

                if self.proxy is not None:
                    if len(sf_terms) > 0 and sf_terms[0] in self.proxy.substitutions:
                        sf_terms[0] += "_nested"
                        search_field = '.'.join(sf_terms)

                search_fields[index] = search_field + '^' + str(index + 1)

            current_qs = _params['body']['query']['query_string']

            if isinstance(current_qs, str):
                _params['body']['query']['query_string'] = {'query': current_qs}
            _params['body']['query']['query_string']['fields'] = search_fields

        return _params

    def do_count(self, params):
        # params['fields'] = []
        params.pop('size', None)
        params.pop('from_', None)
        params.pop('sort', None)
        try:
            return self.api.count(**params)['count']
        except IndexNotFoundException:
            return 0

    def aggregate(self, **params):
        """ Perform aggreration

        Arguments:
            :_aggregations_params: Dict of aggregation params. Root key is an
                aggregation name. Required.
            :_raise_on_empty: Boolean indicating whether to raise exception
                when IndexNotFoundException exception happens. Optional,
                defaults to False.
        """
        _aggregations_params = params.pop('_aggregations_params', None)
        _raise_on_empty = params.pop('_raise_on_empty', False)

        if not _aggregations_params:
            raise Exception('Missing _aggregations_params')

        # Set limit so ES won't complain. It is ignored in the end
        params['_limit'] = 0
        search_params = self.build_search_params(params)
        search_params.pop('size', None)
        search_params.pop('from_', None)
        search_params.pop('sort', None)

        search_params['body']['aggregations'] = _aggregations_params

        log.debug('Performing aggregation: {}'.format(_aggregations_params))
        try:
            response = self.api.search(**search_params)
        except IndexNotFoundException:
            if _raise_on_empty:
                raise JHTTPNotFound(
                    'Aggregation failed: Index does not exist')
            return {}

        try:
            return response['aggregations']
        except KeyError:
            raise JHTTPNotFound('No aggregations returned from ES')

    def get_collection(self, **params):
        _raise_on_empty = params.pop('_raise_on_empty', False)
        _params = self.build_search_params(params)

        if '_count' in params:
            return self.do_count(_params)

        fields = _params.pop('fields', '')
        if fields:
            fields_params = process_fields_param(fields)
            _params.update(fields_params)

        documents = _ESDocs()
        documents._nefertari_meta = dict(
            start=_params.get('from_', 0),
            fields=fields)

        try:
            data = self.api.search(**_params)
        except IndexNotFoundException:
            if _raise_on_empty:
                raise JHTTPNotFound(
                    '{}({}) resource not found (Index does not exist)'.format(
                        self.doc_type, params))
            documents._nefertari_meta.update(
                total=0, took=0)
            return documents

        for found_doc in data['hits']['hits']:
            output_doc = found_doc['_source']
            output_doc['_score'] = found_doc['_score']
            output_doc['_type'] = found_doc['_type']
            documents.append(dict2proxy(output_doc, ES.document_proxies[found_doc['_type']]))

        documents._nefertari_meta.update(
            total=data['hits']['total'],
            took=data['took'],
        )

        if not documents:
            msg = "%s(%s) resource not found" % (self.doc_type, params)
            if _raise_on_empty:
                raise JHTTPNotFound(msg)
            else:
                log.debug(msg)

        return documents

    def get_item(self, **kw):
        _raise_on_empty = kw.pop('_raise_on_empty', True)

        params = dict(
            index=self.index_name,
            doc_type=self.doc_type
        )
        params.update(kw)
        not_found_msg = "'{}({})' resource not found".format(
            self.doc_type, params)

        try:
            data = self.api.get_source(**params)
        except IndexNotFoundException:
            if _raise_on_empty:
                raise JHTTPNotFound("{} (Index does not exist)".format(
                    not_found_msg, self.doc_type, params))
            data = {}
        except JHTTPNotFound:
            data = {}

        if not data:
            if _raise_on_empty:
                raise JHTTPNotFound(not_found_msg)
            else:
                log.debug(not_found_msg)

        if '_type' not in data:
            data['_type'] = self.doc_type

        return dict2proxy(data, self.proxy)

    @classmethod
    def index_relations(cls, db_obj, request=None, **kwargs):
        for model_cls, documents in db_obj.get_related_documents(**kwargs):
            if getattr(model_cls, '_index_enabled', False) and documents:
                children_to_index = []

                for child in documents:
                    pk_name = child.pk_field()
                    if getattr(child, pk_name) is not None:
                        children_to_index.append(child)

                cls(model_cls.__name__).index_documents(children_to_index, request=request)

    @classmethod
    def bulk_index_relations(cls, items, request=None, **kwargs):
        """ Index objects related to :items: in bulk.
        Related items are first grouped in map
        {model_name: {item1, item2, ...}} and then indexed.
        :param items: Sequence of DB objects related objects if which
            should be indexed.
        :param request: Pyramid Request instance.
        """
        index_map = defaultdict(set)

        for item in items:
            relations = item.get_related_documents(**kwargs)
            for model_cls, related_items in relations:
                indexable = getattr(model_cls, '_index_enabled', False)
                if indexable and related_items:
                    index_map[model_cls.__name__].update(related_items)

        for model_name, instances in index_map.items():
            cls(model_name).index_documents(instances, request=request)
