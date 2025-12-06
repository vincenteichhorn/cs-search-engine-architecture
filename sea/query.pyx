from libc.stdint cimport uint32_t, uint64_t, UINT64_MAX
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from sea.query cimport QueryParser, QueryNode
from libc.stdlib cimport malloc, free

cdef void print_query_tree(QueryNode* node, uint64_t depth) noexcept nogil:
    if node == NULL:
        return
    cdef uint64_t i
    for i in range(depth):
        with gil:
            print("  ", end="")
    if node.left == NULL and node.right == NULL:
        if node.values.size() > 1:
            with gil:
                print("Phrase: [", end="")
            for i in range(node.values.size()):
                with gil:
                    print(f"{node.values[i]}", end="")
                if i < node.values.size() - 1:
                    with gil:
                        print(", ", end="")
            with gil:
                print("]")
        else:
            with gil:
                print(f"Token: {node.values[0]}")
    else:
        with gil:
            print(f"Operator: {node.values[0]}")
        print_query_tree(node.left, depth + 1)
        print_query_tree(node.right, depth + 1)

cdef dict query_tree_to_dict(QueryNode* node):
    cdef dict result = {}
    if node == NULL:
        return result
    if node.left == NULL and node.right == NULL:
        if node.values.size() > 1:
            result["type"] = "phrase"
            result["values"] = [node.values[i] for i in range(node.values.size())]
        else:
            result["type"] = "token"
            result["value"] = node.values[0]
    else:
        result["type"] = "operator"
        result["operator"] = node.values[0]
        result["left"] = query_tree_to_dict(node.left)
        result["right"] = query_tree_to_dict(node.right)
    return result

cdef class QueryParser:

    def __cinit__(self, int and_operator, int or_operator, int not_operator, int open_paren, int close_paren, int phrase_marker):

        self.and_operator = <uint64_t>and_operator
        self.or_operator = <uint64_t>or_operator
        self.not_operator = <uint64_t>not_operator
        self.open_paren = <uint64_t>open_paren
        self.close_paren = <uint64_t>close_paren
        self.phrase_marker = <uint64_t>phrase_marker
        self.operator_precedence = unordered_map[uint64_t, uint32_t]()
        self.operator_precedence[self.or_operator] = 1
        self.operator_precedence[self.and_operator] = 2
        self.operator_precedence[self.not_operator] = 3
    
    cpdef dict py_parse(self, list tokens):
        cdef vector[uint64_t] c_tokens = vector[uint64_t]()
        cdef uint64_t token
        cdef int i
        for i in range(len(tokens)):
            token = <uint64_t>tokens[i]
            c_tokens.push_back(token)
        cdef QueryNode* root = self.parse(c_tokens)
        if root == NULL:
            return None
        return query_tree_to_dict(root)
        

    cdef QueryNode* parse(self, vector[uint64_t]& tokens) noexcept nogil:
        if tokens.size() == 0:
            return NULL

        cdef uint64_t token
        cdef uint64_t i
        cdef uint32_t content_token_count = 0
        for i in range(tokens.size()):
            token = tokens[i]
            if (
                token != self.and_operator
                and token != self.or_operator
                and token != self.not_operator
                and token != self.open_paren
                and token != self.close_paren
                and token != self.phrase_marker
            ):
                content_token_count += 1
        if content_token_count == 0:
            return NULL

        self._remove_surrounding_operators(tokens)
        self._remove_consecutive_operators(tokens)
        self._fill_implicit_ands(tokens)
        self._remove_ands_in_phrases(tokens)
        if tokens.size() == 0:
            return NULL
        
        
        cdef vector[uint64_t] operator_stack = vector[uint64_t]()
        cdef vector[QueryNodePtr] value_stack = vector[QueryNodePtr]()
        cdef bint is_phrase = False
        cdef vector[uint64_t] phrase_tokens = vector[uint64_t]()

        cdef uint64_t _operator = UINT64_MAX
        cdef QueryNode* node

        for i in range(tokens.size()):
            token = tokens[i]
            if self.operator_precedence.find(token) != self.operator_precedence.end():
                while (
                    not operator_stack.empty()
                    and self.operator_precedence.find(operator_stack.back()) != self.operator_precedence.end()
                    and self.operator_precedence[operator_stack.back()] > self.operator_precedence[token]
                ):
                    node = new QueryNode()
                    node.values = vector[uint64_t]()
                    node.values.push_back(operator_stack.back())
                    _operator = operator_stack.back()
                    operator_stack.pop_back()
                    node.right = value_stack.back()
                    value_stack.pop_back()
                    node.left = value_stack.back() if _operator != self.not_operator else NULL
                    if _operator != self.not_operator:
                        value_stack.pop_back()
                    value_stack.push_back(node)
                operator_stack.push_back(token)
            elif token == self.open_paren:
                operator_stack.push_back(token)
            elif token == self.close_paren:
                while not operator_stack.empty() and operator_stack.back() != self.open_paren:
                    node = new QueryNode()
                    node.values = vector[uint64_t]()
                    node.values.push_back(operator_stack.back())
                    _operator = operator_stack.back()
                    operator_stack.pop_back()
                    node.right = value_stack.back()
                    value_stack.pop_back()
                    left = value_stack.back() if _operator != self.not_operator else NULL
                    if _operator != self.not_operator:
                        value_stack.pop_back()
                    value_stack.push_back(node)
                operator_stack.pop_back()
            elif token == self.phrase_marker:
                is_phrase = not is_phrase
                if is_phrase:
                    phrase_tokens.swap(vector[uint64_t]())
                else:
                    node = new QueryNode()
                    node.values = phrase_tokens
                    node.left = NULL
                    node.right = NULL
                    value_stack.push_back(node)
            else:
                if is_phrase:
                    phrase_tokens.push_back(token)
                else:
                    node = new QueryNode()
                    node.values = vector[uint64_t]()
                    node.values.push_back(token)
                    node.left = NULL
                    node.right = NULL
                    value_stack.push_back(node)

        while not operator_stack.empty():
            node = new QueryNode()
            node.values = vector[uint64_t]()
            node.values.push_back(operator_stack.back())
            _operator = operator_stack.back()
            operator_stack.pop_back()
            node.right = value_stack.back()
            value_stack.pop_back()
            node.left = value_stack.back() if _operator != self.not_operator else NULL
            if _operator != self.not_operator:
                value_stack.pop_back()
            value_stack.push_back(node)
        return value_stack[0]

    cdef void _remove_surrounding_operators(self, vector[uint64_t]& tokens) noexcept nogil:

        if tokens.size() == 0:
            return

        cdef unordered_set[uint64_t] operators = unordered_set[uint64_t]()
        operators.insert(self.and_operator)
        operators.insert(self.or_operator)
        
        while tokens.size() > 0 and operators.find(tokens[0]) != operators.end():
            tokens.erase(tokens.begin(), tokens.begin() + 1)
        while tokens.size() > 0 and operators.find(tokens[tokens.size() - 1]) != operators.end():
            tokens.pop_back()

    cdef void _remove_consecutive_operators(self, vector[uint64_t]& tokens) noexcept nogil:

        if tokens.size() == 0:
            return

        cdef unordered_set[uint64_t] operators = unordered_set[uint64_t]()
        operators.insert(self.and_operator)
        operators.insert(self.or_operator)

        cdef vector[uint64_t] cleaned_tokens = vector[uint64_t]()
        cdef uint64_t prev_token = UINT64_MAX
        cdef uint64_t token
        cdef uint64_t i

        for i in range(tokens.size()):
            token = tokens[i]
            if operators.find(token) != operators.end() and prev_token != UINT64_MAX and operators.find(prev_token) != operators.end():
                continue
            cleaned_tokens.push_back(token)
            prev_token = token

        tokens.swap(cleaned_tokens)

    cdef void _fill_implicit_ands(self, vector[uint64_t]& tokens) noexcept nogil:

        if tokens.size() == 0:
            return
        
        cdef vector[uint64_t] filled_tokens = vector[uint64_t]()
        filled_tokens.push_back(tokens[0])
        cdef unordered_set[uint64_t] operators = unordered_set[uint64_t]()
        operators.insert(self.and_operator)
        operators.insert(self.or_operator)

        cdef bint is_phrase = False
        cdef uint64_t token
        cdef uint64_t i
        for i in range(1, tokens.size()):
            token = tokens[i]
            if (
                operators.find(token) == operators.end()
                and operators.find(filled_tokens.back()) == operators.end()
                and filled_tokens.back() != self.not_operator
                and not is_phrase
                and filled_tokens.back() != self.open_paren
                and token != self.close_paren
            ):
                filled_tokens.push_back(self.and_operator)
            if token == self.phrase_marker:
                is_phrase = not is_phrase
            filled_tokens.push_back(token)

        tokens.swap(filled_tokens)


    cdef void _remove_ands_in_phrases(self, vector[uint64_t]& tokens) noexcept nogil:

        if tokens.size() == 0:
            return
        
        cdef vector[uint64_t] cleaned_tokens = vector[uint64_t]()
        cdef bint is_phrase = False
        cdef unordered_set[uint64_t] operators = unordered_set[uint64_t]()
        operators.insert(self.and_operator)
        operators.insert(self.or_operator)

        cdef uint64_t token
        cdef uint64_t i
        for i in range(tokens.size()):
            token = tokens[i]
            if token == self.phrase_marker:
                is_phrase = not is_phrase
            if operators.find(token) != operators.end() and is_phrase:
                continue
            cleaned_tokens.push_back(token)
        
        tokens.swap(cleaned_tokens)