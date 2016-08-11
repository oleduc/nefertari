import six
from nefertari.utils.dictset import dictset
from nefertari.utils.utils import issequence


class DocumentView(dictset):
    pass


class DataProxy(object):
    _data = None
    _raw_data = None
    _substituted = None

    def __init__(self, data=None):
        self._data = {}
        self._raw_data = data or {}
        self._substituted = []

        for key, val in data.items():
            setattr(self, key, val)

    def __setattr__(self, name, value):
        """
        Override __setattr__ to replace nested attributes with their backrefs.
        Since we can't have multiple fields with the same names and different mappings and that nested and non nested
        backrefs might have the same name but different mappins, we expand the nested backrefs into:
        Type.backref_name = {type:long}
        Type.backref_name_nested = {*object properties*}
        Which means that objects that are being read from ES need to have backref_name replaced with backref_name_nested
        This is where we do this operation as it is the common denominator of all indexed objects.
        :param name:
        :param value:
        """

        if not name.endswith("_nested"):
            if hasattr(self, "substitutions")and self.substitutions is not None and name in self.substitutions:
                self._substituted.append(name)
                nested_value = self._raw_data[name + "_nested"]

                if isinstance(nested_value, dict):
                    value = dict2obj(nested_value)
                elif isinstance(nested_value, list):
                    value = [dict2obj(sj) if isinstance(sj, dict) else sj for sj in nested_value]
                else:
                    value = nested_value

            if not hasattr(self.__class__, name):
                self._data[name] = value

            super(DataProxy, self).__setattr__(name, value)

    def to_dict(self, **kwargs):
        _dict = dictset()
        _keys = kwargs.pop('_keys', [])
        _depth = kwargs.pop('_depth', 1)

        data = dictset(self._data).subset(_keys) if _keys else self._data

        for attr, val in data.items():
            _dict[attr] = val

            if _depth:
                kw = kwargs.copy()
                kw['_depth'] = _depth - 1

                if hasattr(val, 'to_dict'):
                    _dict[attr] = val.to_dict(**kw)
                elif isinstance(val, list):
                    _dict[attr] = to_dicts(val, **kw)

        return _dict


def dict2obj(data, proxy_cls=None):
    if not data:
        return data

    # Here we create a dynamic type of DataProxy, with the same name as document type.
    # Todo: Make it more consistent with real document instances.
    if proxy_cls is None:
        _type = str(data.get('_type'))
        proxy_cls = type(_type, (DataProxy,), {
            "__init__": DataProxy.__init__,
            "__setattr__": DataProxy.__setattr__,
            "substitutions": None,
            "to_dict": DataProxy.to_dict
        })

    proxy = proxy_cls(data)

    for key, val in proxy._raw_data.items():
        key = str(key)

        if isinstance(val, dict):
            setattr(proxy, key, dict2obj(val))
        elif isinstance(val, list):
            setattr(
                proxy, key,
                [dict2obj(sj) if isinstance(sj, dict) else sj for sj in val])
        else:
            setattr(proxy, key, val)

    return proxy


def dict2proxy(data, proxy):
    return dict2obj(data, proxy)


def to_objs(collection):
    _objs = []

    for each in collection:
        _objs.append(dict2obj(each))

    return _objs


def to_dicts(collection, key=None, **kw):
    _dicts = []
    try:
        for each in collection:
            try:
                each_dict = each.to_dict(**kw)
                if key:
                    each_dict = key(each_dict)
                _dicts.append(each_dict)
            except AttributeError:
                _dicts.append(each)
    except TypeError:
        return collection

    return _dicts


def to_indexable_dicts(collection, key=None, **kw):
    _dicts = []
    try:
        for each in collection:
            try:
                each_dict = each.to_indexable_dict(**kw)
                if key:
                    each_dict = key(each_dict)
                _dicts.append(each_dict)
            except AttributeError:
                _dicts.append(each)
    except TypeError:
        return collection

    return _dicts


def obj2dict(obj, classkey=None):
    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = obj2dict(obj[k], classkey)
        return obj
    elif issequence(obj):
        return [obj2dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dictset([
            (key, obj2dict(value, classkey))
            for key, value in obj.__dict__.items()
            if not six.callable(value) and not key.startswith('_')
        ])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


class FieldData(object):
    """ Keeps field data in a generic format.

    Is passed to field processors.
    """
    def __init__(self, name, new_value, params=None):
        """
        :param name: Name of field.
        :param new_value: New value of field.
        :param params: Dict containing DB field init params.
            E.g. min_length, required.
        """
        self.name = name
        self.new_value = new_value
        self.params = params

    def __repr__(self):
        return '<FieldData: {}>'.format(self.name)

    @classmethod
    def from_dict(cls, data, model):
        """ Generate map of `fieldName: clsInstance` from dict.

        :param data: Dict where keys are field names and values are
            new values of field.
        :param model: Model class to which fields from :data: belong.
        """
        model_provided = model is not None
        result = {}
        for name, new_value in data.items():
            kwargs = {
                'name': name,
                'new_value': new_value,
            }
            if model_provided:
                kwargs['params'] = model.get_field_params(name)
            result[name] = cls(**kwargs)
        return result
