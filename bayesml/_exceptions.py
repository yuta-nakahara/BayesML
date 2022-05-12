# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
class ParameterFormatError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class DataFormatError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class CriteriaError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ResultWarning(UserWarning):
    pass

class ParameterFormatWarning(UserWarning):
    pass
