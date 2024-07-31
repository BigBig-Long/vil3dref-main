# 定义一个函数，用于判断文本中是否明确包含视图相关的词汇
def is_explicitly_view_dependent(tokens):
    """
    :return: 一个布尔类型的掩码
    """
    # 定义一个集合，包含所有视图相关的词汇
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    # 遍历文本中的每个词汇
    for token in tokens:
        # 如果词汇在视图相关词汇集合中，则返回True
        if token in target_words:
            return True
    # 如果没有找到视图相关的词汇，则返回False
    return False
