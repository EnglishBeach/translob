import numpy as _np
import pandas as _pd


class DataClass:
    """
    make only lover case parametrs and not start with _
    All this methods (exept __call__) only for beauty representation :)
    """

    @staticmethod
    def __not_data(field=None, get=False, not_data_fields: set = set()):
        if not get:
            not_data_fields.add(field.__name__)
            return field
        else:
            return not_data_fields

    def __new__(
        cls,
        target_dict: dict = None,
        name: str = '',
    ):
        """
        build from nested dict
        """
        if target_dict is not None:
            result = DataClass()
            return result.__rec_build(name, target_dict)
        return super().__new__(cls)

    def __rec_build(self, field_name: str, field):
        if not isinstance(field, dict):
            self.__setattr__(field_name, field)
            return None

        result = DataClass()
        self.__setattr__(field_name, result)

        for inner_field_name, inner_field in field.items():
            inner_result = result.__rec_build(
                inner_field_name,
                inner_field,
            )
            if inner_result is not None:
                self.__setattr__(field_name, inner_result)
        return result

    def __call__(self, **kwargs: dict):
        """
        Set up parametrs
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __get_all_fields(self):
        # Add except fields

        options = list(
            filter(
                lambda x:
                (x[0] != '_') and (x not in self.__not_data(get=True)),
                self.__dir__(),
            ))
        return options

    def __repr__(self) -> str:
        """
        Representation of options
        """
        return self.__rec_print()[4:]

    def _rec_print_depr(self, self_margin: str = ''):
        if not isinstance(self, DataClass):
            return f'{self}'

        result = self_margin
        for field_name in self.__get_all_fields():
            inner_result = DataClass._rec_print_depr(
                self.__getattribute__(field_name),
                self_margin + ' ' * 4,
            )
            result += f'\n{self_margin}{field_name}: {inner_result}'

        if self_margin == '':
            return result[1:]
        else:
            return result

    def __rec_print(
        self,
        self_name: str = '',
        self_header: str = '',
        last=True,
    ):
        end = "└─ "
        pipe = "│  "
        tee = "├─ "
        blank = "   "
        result = f'{self_header}{end if last else tee}{self_name}\n'

        if not isinstance(self, DataClass):
            if '<' in repr(self):
                self = repr(self).split('at')[0].replace('<', '').strip()

            return f'{self_header}{end if last else tee}{self_name}: {self}\n'

        fields = self.__get_all_fields()
        for field_name in fields:
            inner_result = DataClass.__rec_print(
                self.__getattribute__(field_name),
                self_name=field_name,
                self_header=f'{self_header}{blank if last else pipe}',
                last=field_name == fields[-1])

            result += inner_result[6:]

        return result

    @property
    @__not_data
    def Info_nested(self):
        """
        Containing options dict
        """
        return self.__rec_nested()

    def __rec_nested(self, self_name=None):
        if not isinstance(self, DataClass):
            return {self_name: self}

        result = {}
        for field_name in self.__get_all_fields():
            inner_result = DataClass.__rec_nested(
                self.__getattribute__(field_name),
                field_name,
            )
            result.update(inner_result)

        if self_name is None:
            return result
        else:
            return {self_name: result}

    @property
    @__not_data
    def Info_expanded(self):
        return {
            compound_key.strip()[2:]: value
            for value, compound_key in self.__rec_expanded()
        }

    def __rec_expanded(self, composite_key=''):
        if not isinstance(self, DataClass):
            yield (self, composite_key)
        else:
            for field_name in self.__get_all_fields():
                for inner_result in DataClass.__rec_expanded(
                        self.__getattribute__(field_name),
                        str(composite_key) + '__' + str(field_name),
                ):
                    yield inner_result

    def __getitem__(self, value):
        if isinstance(value, list | tuple):
            result = {}
            for i in value:
                result.update({i: getattr(self, i, None)})
            return DataClass(result)

        result = getattr(self, value, None)
        if isinstance(result, DataClass):
            return DataClass(result.Info_nested)
        else:
            return result


def inspect_data(data, name='data'):
    if data is not None:
        x = data[0]
        y = data[1]
        print(f'{name: <10}: x= {str(x.shape): <15} | y= {str(y.shape): <15}')
    else:
        print(f'{name <10}: None')


def inspect_dataset(ds: _pd.DataFrame, name='dataset'):
    if ds is not None:
        print(f'{name: <10}: {[len(ds)]+ list(ds.element_spec[0].shape)[1:]}')
    else:
        print(f'{name <10}: None')
