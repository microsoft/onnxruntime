# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from opgen.lexer import TokenKind
from opgen.parser import create_from_string as Parser

class ParserLookaheadTests(unittest.TestCase):
  def test_no_peeks(self):
    parser = Parser("1 2 3 4 5")
    self.assertEqual("1", parser._read_token().value)
    self.assertEqual("2", parser._read_token().value)
    self.assertEqual("3", parser._read_token().value)
    self.assertEqual("4", parser._read_token().value)
    self.assertEqual("5", parser._read_token().value)

  def test_peek_0(self):
    parser = Parser("1 2 3 4 5")
    self.assertRaises(IndexError, lambda: parser._peek_token(look_ahead = 0))

  def test_backward_peek_no_reads(self):
    parser = Parser("1 2 3 4 5")
    self.assertEqual("1", parser._peek_token(look_ahead = 1).value)
    self.assertEqual("2", parser._peek_token(look_ahead = 2).value)
    self.assertEqual("3", parser._peek_token(look_ahead = 3).value)
    self.assertEqual("4", parser._peek_token(look_ahead = 4).value)
    self.assertEqual("5", parser._peek_token(look_ahead = 5).value)

  def test_forward_peek_no_reads(self):
    parser = Parser("1 2 3 4 5")
    self.assertEqual("5", parser._peek_token(look_ahead = 5).value)
    self.assertEqual("4", parser._peek_token(look_ahead = 4).value)
    self.assertEqual("3", parser._peek_token(look_ahead = 3).value)
    self.assertEqual("2", parser._peek_token(look_ahead = 2).value)
    self.assertEqual("1", parser._peek_token(look_ahead = 1).value)

  def test_peek_and_read(self):
    parser = Parser("1 2 3 4 5 6 7 8")
    self.assertEqual("1", parser._read_token().value)
    self.assertEqual("2", parser._read_token().value)
    self.assertEqual("4", parser._peek_token(look_ahead = 2).value)
    self.assertEqual("3", parser._peek_token(look_ahead = 1).value)
    self.assertEqual("6", parser._peek_token(look_ahead = 4).value)
    self.assertEqual("3", parser._read_token().value)
    self.assertEqual("4", parser._read_token().value)
    self.assertEqual("5", parser._peek_token(look_ahead = 1).value)
    self.assertEqual("8", parser._peek_token(look_ahead = 4).value)
    self.assertEqual("5", parser._read_token().value)
    self.assertEqual("6", parser._read_token().value)
    self.assertEqual("7", parser._read_token().value)
    self.assertEqual("8", parser._peek_token(look_ahead = 1).value)

  def test_peek_value_and_read(self):
    parser = Parser("1 2 3 4 5 6 7 8")
    self.assertEqual("1", parser._read_token().value)
    self.assertEqual("2", parser._read_token().value)
    self.assertIsNotNone(parser._peek_token(TokenKind.NUMBER, "4", look_ahead = 2))
    self.assertIsNotNone(parser._peek_token(TokenKind.NUMBER, "3", look_ahead = 1))
    self.assertIsNotNone(parser._peek_token(TokenKind.NUMBER, "6", look_ahead = 4))
    self.assertEqual("3", parser._read_token().value)
    self.assertEqual("4", parser._read_token().value)
    self.assertIsNotNone(parser._peek_token(TokenKind.NUMBER, "5", look_ahead = 1))
    self.assertIsNotNone(parser._peek_token(TokenKind.NUMBER, "8", look_ahead = 4))
    self.assertEqual("5", parser._read_token().value)
    self.assertEqual("6", parser._read_token().value)
    self.assertEqual("7", parser._read_token().value)
    self.assertIsNotNone(parser._peek_token(TokenKind.NUMBER, "8", look_ahead = 1))