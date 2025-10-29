from typing import List
from sea.tokenizer import Tokenizer


class Node:

    def __init__(
        self, value: str, is_not: bool = False, left: "Node" = None, right: "Node" = None
    ):  # optionale Argumente
        self.left = left
        self.right = right
        self.value = value
        self.is_not = is_not


class Query:

    def __init__(self, input: str, tokenizer: Tokenizer):
        self.input = input
        self.tokenizer = tokenizer
        self.operator_precedence = {"or": 1, "and": 2}
        self.root = self._parse_query(input)

    def _parse_query(self, input: str) -> Node:
        """
        Parse the input query string into a binary tree structure representing the query.

        Args:
            input (str): The input query string.

        Returns:
            Node: The root node of the binary tree representing the query.
        """
        tokens = self.tokenizer.tokenize(input, is_query=True)
        filled_tokens = self._fill_in_ands(tokens)

        ops = []
        vals = []

        is_prev_not = False
        for token in filled_tokens:
            if token in self.operator_precedence:
                while ops and self.operator_precedence[ops[-1]] >= self.operator_precedence[token]:
                    op = ops.pop()
                    right = vals.pop()
                    left = vals.pop()
                    vals.append(Node(value=op, left=left, right=right))
                ops.append(token)
            elif token != "not":
                vals.append(Node(value=token, is_not=is_prev_not))
            is_prev_not = token == "not"

        while ops:
            op = ops.pop()
            right = vals.pop()
            left = vals.pop()
            vals.append(Node(value=op, left=left, right=right))

        return vals[0]

    def __repr__(self):

        def print_node(node: Node, depth: int = 0) -> str:
            if node is None:
                return ""
            result = "  " * depth
            if node.is_not:
                result += "not "
            result += f"{node.value}\n"
            result += print_node(node.left, depth + 1)
            result += print_node(node.right, depth + 1)
            return result

        return print_node(self.root)

    def _fill_in_ands(self, tokens: List[str]) -> List[str]:
        """
        Fill in implicit AND operators between tokens where no operator is specified.

        Args:
            tokens (List[str]): A list of tokens representing the query.
        Returns:
            List[str]: A list of tokens with implicit AND operators filled in.
        """
        filled_tokens = []
        beginning = True
        operators = {"and", "or"}
        for i, token in enumerate(tokens):
            if token in operators and beginning == True:
                continue
            beginning = False
            next_token = tokens[i + 1] if i + 1 < len(tokens) else "and"
            if token in operators and (
                filled_tokens[-1] == "not" or filled_tokens[-1] in operators
            ):
                continue
            filled_tokens.append(token)
            if token not in operators and (
                token != "not" and next_token not in operators or next_token == "not"
            ):
                filled_tokens.append("and")
        return filled_tokens
