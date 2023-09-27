# FIXME: add graph repr
class DataClass:
    """
    make only lover case parametrs and not start with _
    All this methods (exept __call__) only for beauty representation :)
    """

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
            return

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

    @staticmethod
    def __not_data(field=None, get=False, not_data_fields: set = set()):
        if not get:
            not_data_fields.add(field.__name__)
            return field
        else:
            return not_data_fields

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
        return DataClass.__rec_print(self.Info_nested)

    @staticmethod
    def __rec_print(field, margin: str = ''):
        if not isinstance(field, dict):
            return f'{field}'
        result = margin

        for field_name, field in field.items():
            inner_result = DataClass.__rec_print(
                field,
                margin + ' ' * 4,
            )
            result += f'\n{margin}{field_name}: {inner_result}'

        if margin == '':
            return result[1:]
        else:
            return result

    @property
    @__not_data
    def Info_nested(self):
        """
        Containing options dict
        """
        return self.__rec_dict()

    def __rec_dict(self, self_name=None):
        if not isinstance(self, DataClass):
            return {self_name: self}

        result = {}
        for field_name in self.__get_all_fields():

            inner_result = DataClass.__rec_dict(
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
            for value, compound_key in self.__rec_grid()
        }

    def __rec_grid(self, composite_key=''):
        if not isinstance(self, DataClass):
            yield (self, composite_key)
        else:
            for field_name in self.__get_all_fields():
                for inner_result in DataClass.__rec_grid(
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
