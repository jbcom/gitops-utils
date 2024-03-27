import re

IS_MAGIC_METHOD = re.compile(r"^__.*__$")
IS_MULTILINE_CONDITIONAL_EXPR = re.compile(r"\n|\|\||&&")
IS_NON_WORD_CHAR = re.compile(r"\W")
