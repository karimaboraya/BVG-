class Station:

    def __init__(self, name, time, line, cor_x, cor_y, change):

        self._name = name
        self._time = time
        self._line = []
        self._cor_x = cor_x
        self._cor_y = cor_y
        self._change = change

    def get_name(self):
        return self._name

    def get_line(self):
        return self._line

    def get_time(self):
        return self._time

    def get_cor_x(self):
        return self._cor_x

    def get_cor_y(self):
        return self._cor_y

    def get_change(self):
        return self._change

    def set_name(self, name):
        self._name = name

    def set_line(self, line_name):
        self._line = line_name

    def set_time(self, time):
        self._time = time

    def set_cor_x(self, cor_x):
        self._cor_x = cor_x

    def set_cor_y(self, cor_y):
        self._cor_y = cor_y

    def set_change(self, change):
        self._change = change

    def __str__(self):
        return "\n Station: " + self._name + "\n Line: " + self._line

    def l_sort(self, Stations):
        return self._time


class line(Station):
    pass

