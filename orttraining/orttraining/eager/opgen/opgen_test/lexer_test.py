# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from opgen.lexer import StringReader, Lexer, Token, TokenKind, SourceLocation


class LexerTestCase(unittest.TestCase):
    def create_lexer(self, buffer: str) -> Lexer:
        return Lexer(StringReader(buffer))

    def lex_single(self, buffer: str, expected_kind: TokenKind):
        lexer = self.create_lexer(buffer)
        token = lexer.lex()
        self.assertEqual(expected_kind, token.kind)
        self.assertIsNone(token.leading_trivia)
        self.assertIsNone(token.trailing_trivia)
        eof = lexer.lex()
        self.assertEqual(TokenKind.EOF, eof.kind)
        self.assertIsNone(eof.value)
        self.assertIsNone(eof.leading_trivia)
        self.assertIsNone(eof.trailing_trivia)
        return token

    def test_empty(self):
        lexer = self.create_lexer("")
        self.assertEqual(lexer.lex().kind, TokenKind.EOF)

    def test_trivia(self):
        lexer = self.create_lexer("    // hello\nid\r\n// world")
        self.assertEqual(
            Token(
                (13, 2, 1),
                TokenKind.IDENTIFIER,
                "id",
                leading_trivia=[
                    Token((0, 1, 1), TokenKind.WHITESPACE, "    "),
                    Token((4, 1, 5), TokenKind.SINGLE_LINE_COMMENT, "// hello"),
                    Token((12, 1, 13), TokenKind.WHITESPACE, "\n"),
                ],
                trailing_trivia=[
                    Token((15, 2, 3), TokenKind.WHITESPACE, "\r\n"),
                    Token((17, 3, 1), TokenKind.SINGLE_LINE_COMMENT, "// world"),
                ],
            ),
            lexer.lex(),
        )
        self.assertEqual(Token((25, 3, 9), TokenKind.EOF, None), lexer.lex())

    def test_number(self):
        def assert_number(number):
            self.assertEqual(number, self.lex_single(number, TokenKind.NUMBER).value)

        for number in [
            "0",
            "01",
            "-1",
            "-.5",
            ".5",
            "0.5",
            "0e1",
            "1E10",
            "-0.5e6",
            "-0.123e-456",
            "1234567891011231314151617181920",
            "-12345.6789E-123456",
        ]:
            assert_number(number)

        for number in ["1.2.3", "e1", "-e1", "123e0.5"]:
            self.assertRaises(BaseException, lambda: assert_number(number))

        lexer = self.create_lexer("1.2.3.4e5.6")
        self.assertEqual(Token((0, 1, 1), TokenKind.NUMBER, "1.2"), lexer.lex())
        self.assertEqual(Token((3, 1, 4), TokenKind.NUMBER, ".3"), lexer.lex())
        self.assertEqual(Token((5, 1, 6), TokenKind.NUMBER, ".4e5"), lexer.lex())
        self.assertEqual(Token((9, 1, 10), TokenKind.NUMBER, ".6"), lexer.lex())
        self.assertEqual(Token((11, 1, 12), TokenKind.EOF, None), lexer.lex())
