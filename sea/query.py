from typing import List, Union
from sea.tokenizer import Tokenizer


class Node:

    def __init__(
        self,
        value: Union[List[str], str],
        left: Union["Node", None] = None,
        right: Union["Node", None] = None,
    ):
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):

        def print_node(node: Union["Node", None], depth: int = 0) -> str:
            if node is None:
                return ""
            result = "  " * depth
            result += f"{node.value}\n"
            result += print_node(node.left, depth + 1)
            result += print_node(node.right, depth + 1)
            return result

        return print_node(self)


class Query:

    def __init__(self, input: str, tokenizer: Tokenizer):
        self.input = input
        self.tokenizer = tokenizer
        self.operator_precedence = {"or": 1, "and": 2, "not": 3}
        self.root = self._parse_query(input)

    def _parse_query(self, input: str) -> Union[Node, None]:
        """
        Parse the input query string into a binary tree structure representing the query.

        Args:
            input (str): The input query string.

        Returns:
            Node: The root node of the binary tree representing the query.
        """
        tokens = self.tokenizer.tokenize(input, is_query=True)

        if tokens == []:
            return None
        number_tokens = 0
        for token in tokens:
            if token not in self.operator_precedence.keys() and token != '"':
                number_tokens += 1
        if number_tokens == 0:
            return None

        filled_tokens = self._remove_surrounding_operators(tokens)
        filled_tokens = self._remove_consecutive_operators(filled_tokens)
        filled_tokens = self._fill_in_implicit_ands(filled_tokens)
        filled_tokens = self._remove_in_phrase_ands(filled_tokens)

        if filled_tokens == []:
            return None

        ops = []
        vals = []

        is_phrase = False
        phrase_tokens = []
        for token in filled_tokens:
            if token in self.operator_precedence:
                while (
                    ops
                    and ops[-1] in self.operator_precedence
                    and self.operator_precedence[ops[-1]] > self.operator_precedence[token]
                ):
                    op = ops.pop()
                    right = vals.pop()
                    left = vals.pop() if op != "not" else None
                    vals.append(Node(value=op, left=left, right=right))
                ops.append(token)
            elif token == "(":
                ops.append(token)
            elif token == ")":
                while ops and ops[-1] != "(":
                    op = ops.pop()
                    right = vals.pop()
                    left = vals.pop() if op != "not" else None
                    vals.append(Node(value=op, left=left, right=right))

                ops.pop()
            elif token == '"':
                is_phrase = not is_phrase
                if is_phrase:
                    phrase_tokens = []
                else:
                    vals.append(Node(value=phrase_tokens))
            else:
                if is_phrase:
                    phrase_tokens.append(token)
                else:
                    vals.append(Node(value=token))
        while ops:
            op = ops.pop()
            right = vals.pop()
            left = vals.pop() if op != "not" else None
            vals.append(Node(value=op, left=left, right=right))

        return vals[0]

    def __repr__(self):
        return self.root.__repr__()

    def _remove_surrounding_operators(self, tokens: List[str]) -> List[str]:
        """
        Remove leading and trailing operators from the token list.

        Args:
            tokens (List[str]): A list of tokens representing the query.
        Returns:
            List[str]: A list of tokens with leading and trailing operators removed.
        """
        operators = {"and", "or"}
        first_valid_token_index = 0
        last_valid_token_index = len(tokens) - 1
        while tokens[first_valid_token_index] in operators:
            first_valid_token_index += 1
        while tokens[last_valid_token_index] in operators:
            last_valid_token_index -= 1
        return tokens[first_valid_token_index : last_valid_token_index + 1]

    def _remove_consecutive_operators(self, tokens: List[str]) -> List[str]:
        """
        Remove consecutive operators from the token list.

        Args:
            tokens (List[str]): A list of tokens representing the query.
        Returns:
            List[str]: A list of tokens with consecutive operators removed.
        """
        if not tokens:
            return tokens

        cleaned_tokens = [tokens[0]]
        operators = {"and", "or"}

        for token in tokens[1:]:
            if token in operators and cleaned_tokens[-1] in operators:
                continue
            cleaned_tokens.append(token)

        return cleaned_tokens

    def _fill_in_implicit_ands(self, tokens: List[str]) -> List[str]:
        """
        Fill in implicit AND operators between tokens where no operator is specified.

        Args:
            tokens (List[str]): A list of tokens representing the query.
        Returns:
            List[str]: A list of tokens with implicit AND operators filled in.
        """
        if not tokens:
            return tokens
        filled_tokens = [tokens[0]]
        operators = {"and", "or"}
        is_phrase = False
        for token in tokens[1:]:
            if (
                token not in operators
                and filled_tokens[-1] not in operators
                and filled_tokens[-1] != "not"
                and not is_phrase
                and filled_tokens[-1] != "("
                and token != ")"
            ):
                filled_tokens.append("and")
            if token == '"':
                is_phrase = not is_phrase
            filled_tokens.append(token)
        return filled_tokens

    def _remove_in_phrase_ands(self, tokens: List[str]) -> List[str]:
        """
        Remove 'and' operators that are inside phrases (between quotation marks).

        Args:
            tokens (List[str]): A list of tokens representing the query.
        Returns:
            List[str]: A list of tokens with 'and' operators inside phrases removed.
        """
        cleaned_tokens = []
        is_phrase = False
        operators = {"and", "or"}
        for token in tokens:
            if token == '"':
                is_phrase = not is_phrase
            if token in operators and is_phrase:
                continue
            cleaned_tokens.append(token)
        return cleaned_tokens
