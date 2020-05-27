"""
本代码为树结构的基本类
"""


class Tree:
    def __init__(self, val='root', children=None):
        self.val = val
        self.children = {}
        if children:
            for child in children:
                assert isinstance(child, Tree)
                self.children[child.val] = child

        return

    def add_child(self, child):
        assert isinstance(child, Tree)
        self.children[child.val] = child

        return

    def get_child(self, name):
        return self.children[name]

