class Tree:
    def __init__(self, attribute, split_val, level):
        self.attribute = attribute
        self.split_val = split_val
        self.level = level
        self.children = []
    
    def add_child_node(self, child):
        self.children.append(child)

