class DataClass:
    """
    make only lover case parametrs and not start with _
    All this methods (exept __call__) only for beauty representation :)
    """
    _data_filter = lambda x: (x[0] != '_') and ('Data' not in x)

    def __new__(
        cls,
        target_dict: dict = None,
        name: str = '',
    ):
        if target_dict is not None:
            result = DataClass()
            return result._rec_build_(name, target_dict)
        return super().__new__(cls)

    def __call__(self, **kwargs: dict):
        """
        Set up parametrs
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __repr__(self) -> str:
        """
        Representation of options
        """
        return DataClass._rec_print_(self.Data)

    @property
    def Data(self):
        """
        Containing options dict
        """
        return self._rec_dict_()

    def _get_all_options(self):
        # Add except fields

        options = list(filter(DataClass._data_filter, self.__dir__()))
        return options

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
