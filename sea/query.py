from typing import List
from sea.tokenizer import Tokenizer


class Node:

    def __init__(self, value: str, left: "Node" = None, right: "Node" = None):
        self.left = left
        self.right = right
        self.value = value


class Query:

    def __init__(self, input: str, tokenizer: Tokenizer):
        self.input = input
        self.tokenizer = tokenizer
        self.operator_precedence = {"or": 1, "and": 2, "not": 3}
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
        filled_tokens = tokens  # self._fill_in_ands(tokens)
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
                    left = vals.pop()
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

        def print_node(node: Node, depth: int = 0) -> str:
            if node is None:
                return ""
            result = "  " * depth
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

        TODO: Handle edge cases more gracefully. REFACTOR!
            - not not and not tree
        TOTAL_HOURS_WASTED_HERE = 1
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

        end = True
        for i, token in enumerate(reversed(filled_tokens)):
            if (token in operators or token == "not") and end == True:
                filled_tokens.remove(token)
            else:
                end = False

        return filled_tokens


if __name__ == "__main__":
    tokenizer = Tokenizer()
    # query = 'not tree and ( rock or water ) and "new york city"'
    # print(query)
    # q = Query(query, tokenizer)
    # print(q)

    query = 'not not ( tree and rock ) or water and "new york city"'
    print(query)
    q = Query(query, tokenizer)
    print(q)

    # query = "not not root and not tree"
    # print(query)
    # q = Query(query, tokenizer)
    # print(q)

    # query = 'not tree and rock or water and "new york city"'
    # print(query)
    # q = Query(query, tokenizer)
    # print(q)
