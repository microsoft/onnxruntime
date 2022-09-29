# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from enum import Enum
from abc import ABC
from typing import List, Optional, Union, Tuple


class SourceLocation(object):
    def __init__(self, offset: int = 0, line: int = 1, column: int = 1):
        self.offset = offset
        self.line = line
        self.column = column

    def increment_line(self):
        return SourceLocation(self.offset + 1, self.line + 1, 1)

    def increment_column(self, count: int = 1):
        return SourceLocation(self.offset + count, self.line, self.column + count)

    def __str__(self) -> str:
        return f"({self.line},{self.column})"

    def __repr__(self) -> str:
        return f"({self.offset},{self.line},{self.column})"

    def __eq__(self, other) -> bool:
        return (
            self.__class__ == other.__class__
            and self.offset == other.offset
            and self.line == other.line
            and self.column == other.column
        )


class TokenKind(Enum):
    UNKNOWN = 1
    EOF = 2
    WHITESPACE = 3
    SINGLE_LINE_COMMENT = 4
    MULTI_LINE_COMMENT = 5
    IDENTIFIER = 6
    NUMBER = 7
    STRING = 8
    OPEN_PAREN = 9
    CLOSE_PAREN = 10
    OPEN_BRACKET = 11
    CLOSE_BRACKET = 12
    LESS_THAN = 13
    GREATER_THAN = 14
    COMMA = 15
    SEMICOLON = 16
    COLON = 17
    DOUBLECOLON = 18
    AND = 19
    OR = 20
    DIV = 21
    MUL = 22
    MINUS = 23
    EQUALS = 24
    QUESTION_MARK = 25
    EXCLAIMATION_MARK = 26
    ARROW = 27


class Token(object):
    def __init__(
        self,
        location: Union[SourceLocation, Tuple[int, int, int]],
        kind: TokenKind,
        value: str,
        leading_trivia: Optional[List["Token"]] = None,
        trailing_trivia: Optional[List["Token"]] = None,
    ):
        if isinstance(location, tuple) or isinstance(location, list):
            location = SourceLocation(location[0], location[1], location[2])

        self.location = location
        self.kind = kind
        self.value = value
        self.leading_trivia = leading_trivia
        self.trailing_trivia = trailing_trivia

    def is_trivia(self) -> bool:
        return (
            self.kind == TokenKind.WHITESPACE
            or self.kind == TokenKind.SINGLE_LINE_COMMENT
            or self.kind == TokenKind.MULTI_LINE_COMMENT
        )

    def has_trailing_trivia(self, trivia_kind: TokenKind) -> bool:
        if not self.trailing_trivia:
            return False
        for trivia in self.trailing_trivia:
            if trivia.kind == trivia_kind:
                return True
        return False

    def __str__(self) -> str:
        return f"{self.location}: [{self.kind}] '{self.value}'"

    def __repr__(self) -> str:
        rep = f"Token({repr(self.location)},{self.kind}"
        if self.value:
            rep += ',"' + self.value + '"'
        if self.leading_trivia:
            rep += f",leading_trivia={self.leading_trivia}"
        if self.trailing_trivia:
            rep += f",trailing_trivia={self.trailing_trivia}"
        return rep + ")"

    def __eq__(self, other) -> bool:
        return (
            self.__class__ == other.__class__
            and self.location == other.location
            and self.kind == other.kind
            and self.value == other.value
            and self.leading_trivia == other.leading_trivia
            and self.trailing_trivia == other.trailing_trivia
        )


class Reader(ABC):
    def open(self):
        pass

    def close(self):
        pass

    def read_char(self) -> str:
        return None


class FileReader(Reader):
    def __init__(self, path: str):
        self.path = path

    def open(self):
        self.fp = open(self.path)

    def close(self):
        self.fp.close()

    def read_char(self) -> str:
        return self.fp.read(1)


class StringReader(Reader):
    def __init__(self, buffer: str):
        self.buffer = buffer
        self.position = 0

    def read_char(self) -> str:
        if self.position < len(self.buffer):
            c = self.buffer[self.position]
            self.position += 1
            return c
        return None


class Lexer(object):
    _peek: str
    _next_token: Token
    _first_token_leading_trivia: List[Token]

    char_to_token_kind = {
        "(": TokenKind.OPEN_PAREN,
        ")": TokenKind.CLOSE_PAREN,
        "<": TokenKind.LESS_THAN,
        ">": TokenKind.GREATER_THAN,
        "[": TokenKind.OPEN_BRACKET,
        "]": TokenKind.CLOSE_BRACKET,
        ",": TokenKind.COMMA,
        ";": TokenKind.SEMICOLON,
        "&": TokenKind.AND,
        "|": TokenKind.OR,
        "=": TokenKind.EQUALS,
        "?": TokenKind.QUESTION_MARK,
        "!": TokenKind.EXCLAIMATION_MARK,
        "*": TokenKind.MUL,
    }

    def __init__(self, reader: Reader):
        self._reader = reader
        self._peek = None
        self._current_token_location = SourceLocation()
        self._next_token_location = SourceLocation()
        self._next_token = None
        self._first_token_leading_trivia = []

    def __enter__(self):
        self._reader.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._reader.close()

    def _make_token(self, kind: TokenKind, value: str) -> Token:
        return Token(self._next_token_location, kind, value)

    def _peek_char(self) -> str:
        if self._peek:
            return self._peek
        self._peek = self._reader.read_char()
        return self._peek

    def _read_char(self) -> str:
        if self._peek:
            c = self._peek
            self._peek = None
        else:
            c = self._reader.read_char()
        if c:
            self._current_token_location = (
                self._current_token_location.increment_line()
                if c == "\n"
                else self._current_token_location.increment_column()
            )
        return c

    def set_source_location(self, origin: SourceLocation):
        self._current_token_location = origin

    def lex(self) -> Token:
        """
        Lex a single semantic token from the source, gathering into it
        any trailing whitespace or comment trivia that may follow. The
        first non-trivia token in the buffer may also have leading trivia
        attached to it.
        """
        token: Token
        leading_trivia: Optional[List[Token]] = None
        trailing_trivia: Optional[List[Token]] = None

        while True:
            token = self._lex_core()
            if token.is_trivia():
                if not leading_trivia:
                    leading_trivia = [token]
                else:
                    leading_trivia.append(token)
            else:
                break

        while True:
            trailing = self._lex_core()
            if trailing.is_trivia():
                if not trailing_trivia:
                    trailing_trivia = [trailing]
                else:
                    trailing_trivia.append(trailing)
            else:
                self._next_token = trailing
                break

        token.leading_trivia = leading_trivia
        token.trailing_trivia = trailing_trivia

        return token

    def _lex_core(self) -> Token:
        """Lex a single token from the source including comments and whitespace."""
        if self._next_token:
            token = self._next_token
            self._next_token = None
            return token

        self._next_token_location = self._current_token_location

        c = self._peek_char()
        if not c:
            return self._make_token(TokenKind.EOF, None)

        kind = Lexer.char_to_token_kind.get(c)
        if kind:
            return self._make_token(kind, self._read_char())

        if c.isspace():
            return self._lex_sequence(TokenKind.WHITESPACE, lambda c: c.isspace())

        if self._is_identifier_char(c, first_char=True):
            return self._lex_sequence(TokenKind.IDENTIFIER, lambda c: self._is_identifier_char(c))

        if c == ":":
            self._read_char()
            if self._peek_char() == ":":
                return self._make_token(TokenKind.DOUBLECOLON, c + self._read_char())
            else:
                return self._make_token(TokenKind.COLON, c)

        if c == "/":
            self._read_char()
            if self._peek_char() == "/":
                return self._lex_sequence(TokenKind.SINGLE_LINE_COMMENT, lambda c: c != "\n", "/")
            elif self._peek_char() == "*":
                raise NotImplementedError("Multi-line comments not supported")
            else:
                return self._make_token(TokenKind.DIV, c)

        if c == "-":
            self._read_char()
            p = self._peek_char()
            if p == ">":
                return self._make_token(TokenKind.ARROW, c + self._read_char())
            elif p == "." or p.isnumeric():
                return self._lex_number(c)
            else:
                return self._make_token(TokenKind.MINUS, c)

        if c == "." or c.isnumeric():
            return self._lex_number()

        if c == '"' or c == "'":
            return self._lex_string()

        return self._make_token(TokenKind.UNKNOWN, c)

    def _lex_number(self, s: str = "") -> Token:
        s += self._read_char()

        in_exponent = False
        have_decimal_separator = "." in s

        while True:
            p = self._peek_char()
            if not p:
                break
            if p.isnumeric():
                s += self._read_char()
            elif not have_decimal_separator and not in_exponent and p == ".":
                have_decimal_separator = True
                s += self._read_char()
            elif not in_exponent and (p == "e" or p == "E"):
                in_exponent = True
                s += self._read_char()
                if self._peek_char() == "-":
                    s += self._read_char()
            else:
                break

        return self._make_token(TokenKind.NUMBER, s)

    def _lex_string(self) -> Token:
        term = self._read_char()
        s = ""
        while True:
            p = self._peek_char()
            if p == "\\":
                self._read_char()
                s += self._read_char()
            elif p == term:
                self._read_char()
                return self._make_token(TokenKind.STRING, s)
            else:
                s += self._read_char()

    def _is_identifier_char(self, c: str, first_char=False) -> bool:
        if c == "_" or c.isalpha():
            return True
        return c in [":", "."] or c.isnumeric() if not first_char else False

    def _lex_sequence(self, kind: TokenKind, predicate, s: str = ""):
        while True:
            c = self._read_char()
            if c:
                s += c
            p = self._peek_char()
            if not p or not predicate(p):
                return self._make_token(kind, s)
