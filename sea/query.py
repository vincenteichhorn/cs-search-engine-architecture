

from typing import List
from sea.tokenizer import Tokenizer


class Node:
    
    def __init__(self, value: str, left: "Node" = None, right: "Node" = None): #optionale Argumente
        self.left = left
        self.right = right
        self.value = value


class Query:

    def __init__(self, input: str, tokenizer: Tokenizer):
        self.input = input
        self.tokenizer = tokenizer
        self.root = self.parse_query(input)

    def parse_query(self, input: str) -> Node:
        tokens = self.tokenizer.tokenize(input, is_query=True)
        filled_tokens = self._fill_in_ands(tokens)
        print(filled_tokens)

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
            print (f"token: {token}, next_token: {next_token}")
            print (token not in operators and next_token not in operators)
            if token in operators and (filled_tokens[-1] == "not" or filled_tokens[-1] in operators):
                continue
            filled_tokens.append(token)
            if token not in operators and token != "not" and next_token not in operators or token not in operators and next_token == "not":
                filled_tokens.append("and")
        return filled_tokens


if __name__ == "__main__":
    tokenizer = Tokenizer()
    query = Query("apple and and banana", tokenizer)
    query = Query("not and apple and banana or cherry", tokenizer)
    query = Query("tree and not house", tokenizer)
    query = Query("house not apple and banana or cherry", tokenizer)
    query = Query("tree or not cherry", tokenizer)