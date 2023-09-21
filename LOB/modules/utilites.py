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
        if target_dict is not None:
            result = DataClass()
            return result._rec_build_(name, target_dict)
        return super().__new__(cls)

    def _rec_build_(self, self_name: str, target):
        if not isinstance(target, dict):
            self.__setattr__(self_name, target)
            return

        result = DataClass()
        self.__setattr__(self_name, result)

        for name, inner_target in target.items():
            inner_result = result._rec_build_(
                name,
                inner_target,
            )
            if inner_result is not None:
                self.__setattr__(self_name, inner_result)
        return result

    @staticmethod
    def _not_data(field=None, get=False, not_data_fields: set = set()):
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

    def _get_all_options(self):
        # Add except fields

        options = list(
            filter(
                lambda x:
                (x[0] != '_') and (x not in self._not_data(get=True)),
                self.__dir__(),
            ))
        return options

    def __repr__(self) -> str:
        """
        Representation of options
        """
        return DataClass._rec_print_(self.Info_nested)

    @staticmethod
    def _rec_print_(target, margin: str = ''):
        if not isinstance(target, dict):
            return f'{target}'
        result = margin

        for key, value in target.items():
            inner_string = DataClass._rec_print_(
                value,
                margin + ' ' * 4,
            )
            result += '\n' + margin + f'{key}: {inner_string}'

        if margin == '':
            return result[1:]
        else:
            return result

    @property
    @_not_data
    def Info_nested(self):
        """
        Containing options dict
        """
        return self._rec_dict_()

    def _rec_dict_(self, self_name=None):
        if not isinstance(self, DataClass):
            return {self_name: self}

        result = {}
        for option in self._get_all_options():

            inner_dict = DataClass._rec_dict_(
                self.__getattribute__(option),
                option,
            )
            result.update(inner_dict)

        if self_name is None:
            return result
        else:
            return {self_name: result}

    @property
    @_not_data
    def Info_expanded(self):
        return {
            compound_key.strip()[2:]: value
            for value, compound_key in self._rec_grid_()
        }

    def _rec_grid_(self, composite_key=''):
        if not isinstance(self, DataClass):
            yield (self, composite_key)
        else:
            for key in self._get_all_options():
                for inner_key in DataClass._rec_grid_(
                        self.__getattribute__(key),
                        str(composite_key) + '__' + str(key),
                ):
                    yield inner_key

    def __getitem__(self, value):
        return getattr(self, value, None)
