class Tree:
    def __init__(self, attribute, parent_attribute, parent_val, median, cat_val_list, child_list, label, level):
        self.attribute = attribute
        self.parent_attribute = parent_attribute
        self.parent_val = parent_val
        self.median = median
        self.cat_val_list = cat_val_list
        self.child_list = child_list
        self.label = label
        self.level = level

    
    def add_child_node(self, child):
        self.child_list.append(child)

