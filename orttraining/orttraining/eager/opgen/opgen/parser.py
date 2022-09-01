# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from opgen.lexer import *
from opgen.ast import *
from typing import List, Tuple, Union, Optional


class UnexpectedTokenError(RuntimeError):
    def __init__(self, expected: TokenKind, actual: Token):
        self.expected = expected
        self.actual = actual
        super().__init__(f"unexpected token {actual}; expected {expected}")


class ExpectedSyntaxError(RuntimeError):
    def __init__(self, expected: str, actual: Token = None):
        super().__init__(f"expected {expected}; actual {actual}")


class ParserBase(object):
    _peek_queue: List[Token]

    def __init__(self, lexer: Union[Lexer, Reader]):
        self._own_lexer = False
        if isinstance(lexer, Reader):
            self._own_lexer = True
            lexer = Lexer(lexer)
        elif not isinstance(lexer, Lexer):
            raise TypeError("lexer must be a Lexer or Reader")
        self._lexer = lexer
        self._peek_queue = []

    def __enter__(self):
        if self._own_lexer:
            self._lexer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._own_lexer:
            self._lexer.__exit__(exc_type, exc_val, exc_tb)

    def set_source_location(self, origin: SourceLocation):
        self._lexer.set_source_location(origin)

    def _peek_token(
        self, kinds: Union[TokenKind, List[TokenKind]] = None, value: str = None, look_ahead: int = 1
    ) -> Optional[Token]:
        if look_ahead < 1:
            raise IndexError("look_ahead must be at least 1")
        if look_ahead >= len(self._peek_queue):
            for _ in range(look_ahead - len(self._peek_queue)):
                self._peek_queue = [self._lexer.lex()] + self._peek_queue
        peek = self._peek_queue[-look_ahead]
        if not kinds:
            return peek
        if not isinstance(kinds, list):
            kinds = [kinds]
        for kind in kinds:
            if peek.kind == kind:
                if value:
                    return peek if peek.value == value else None
                return peek
        return None

    def _read_token(self) -> Token:
        return self._peek_queue.pop() if self._peek_queue else self._lexer.lex()

    def _expect_token(self, kind: TokenKind) -> Token:
        token = self._read_token()
        if token.kind != kind:
            raise UnexpectedTokenError(kind, token)
        return token

    def _parse_list(
        self,
        open_token_kind: TokenKind,
        separator_token_kind: TokenKind,
        close_token_kind: TokenKind,
        member_parser: callable,
    ) -> SyntaxList:
        syntax_list = SyntaxList()
        if open_token_kind:
            syntax_list.open_token = self._expect_token(open_token_kind)
        while True:
            if close_token_kind and self._peek_token(close_token_kind):
                break
            member = member_parser()
            if not self._peek_token(separator_token_kind):
                syntax_list.append(member, None)
                break
            syntax_list.append(member, self._read_token())
        if close_token_kind:
            syntax_list.close_token = self._expect_token(close_token_kind)
        return syntax_list

    def parse_translation_unit(self) -> TranslationUnitDecl:
        decls = []
        while not self._peek_token(TokenKind.EOF):
            decls.append(self.parse_function())
        return TranslationUnitDecl(decls)

    def parse_function_parameter_default_value_expression(self) -> Expression:
        return self.parse_expression()

    def parse_function_parameter(self) -> ParameterDecl:
        parameter_type = self.parse_type()

        if not self._peek_token(TokenKind.IDENTIFIER):
            return ParameterDecl(parameter_type)

        parameter_name = self._read_token()

        if not self._peek_token(TokenKind.EQUALS):
            return ParameterDecl(parameter_type, parameter_name)

        return ParameterDecl(
            parameter_type, parameter_name, self._read_token(), self.parse_function_parameter_default_value_expression()
        )

    def parse_function_parameters(self) -> SyntaxList:
        return self._parse_list(
            TokenKind.OPEN_PAREN, TokenKind.COMMA, TokenKind.CLOSE_PAREN, self.parse_function_parameter
        )

    def parse_function(self) -> FunctionDecl:
        raise NotImplementedError()

    def parse_expression(self) -> Expression:
        raise NotImplementedError()

    def parse_type(self) -> Type:
        raise NotImplementedError()


class CPPParser(ParserBase):
    def parse_function(self) -> FunctionDecl:
        return_type = self.parse_type()
        return FunctionDecl(
            self._expect_token(TokenKind.IDENTIFIER),
            self.parse_function_parameters(),
            return_type,
            semicolon=self._expect_token(TokenKind.SEMICOLON),
        )

    def parse_expression(self) -> Expression:
        if (
            self._peek_token(TokenKind.IDENTIFIER)
            or self._peek_token(TokenKind.NUMBER)
            or self._peek_token(TokenKind.STRING)
        ):
            return LiteralExpression(self._read_token())
        else:
            raise UnexpectedTokenError("expression", self._peek_token())

    def parse_type(self) -> Type:
        if self._peek_token(TokenKind.IDENTIFIER, "const"):
            parsed_type = ConstType(self._read_token(), self.parse_type())
        elif self._peek_token([TokenKind.IDENTIFIER, TokenKind.DOUBLECOLON]):
            identifiers = []
            while True:
                token = self._peek_token([TokenKind.IDENTIFIER, TokenKind.DOUBLECOLON])
                if not token:
                    break
                identifiers.append(self._read_token())
                if token.has_trailing_trivia(TokenKind.WHITESPACE):
                    break
            if self._peek_token(TokenKind.LESS_THAN):
                parsed_type = TemplateType(
                    identifiers,
                    self._parse_list(
                        TokenKind.LESS_THAN, TokenKind.COMMA, TokenKind.GREATER_THAN, self._parse_template_type_argument
                    ),
                )
            elif identifiers[-1].value == "TensorOptions":
                parsed_type = TensorOptionsType(identifiers)
            else:
                parsed_type = ConcreteType(identifiers)
        else:
            raise ExpectedSyntaxError("type", self._peek_token())

        while True:
            if self._peek_token(TokenKind.AND):
                parsed_type = ReferenceType(parsed_type, self._read_token())
            else:
                return parsed_type

    def _parse_template_type_argument(self) -> Type:
        if self._peek_token(TokenKind.NUMBER):
            return ExpressionType(self.parse_expression())
        return self.parse_type()


class TorchParser(ParserBase):
    def __init__(self, lexer: Union[Lexer, Reader]):
        super().__init__(lexer)
        self._next_anonymous_alias_id = 0

    def parse_function(self) -> FunctionDecl:
        return FunctionDecl(
            self._expect_token(TokenKind.IDENTIFIER),
            self.parse_function_parameters(),
            arrow=self._expect_token(TokenKind.ARROW),
            return_type=self.parse_type(),
        )

    def parse_expression(self) -> Expression:
        if (
            self._peek_token(TokenKind.NUMBER)
            or self._peek_token(TokenKind.IDENTIFIER)
            or self._peek_token(TokenKind.STRING)
        ):
            return LiteralExpression(self._read_token())
        elif self._peek_token(TokenKind.OPEN_BRACKET):
            return ArrayExpression(
                self._parse_list(
                    TokenKind.OPEN_BRACKET, TokenKind.COMMA, TokenKind.CLOSE_BRACKET, self.parse_expression
                )
            )
        else:
            raise UnexpectedTokenError("expression", self._peek_token())

    def _create_alias_info_type(self, parsed_type: Type, alias_info: AliasInfo) -> AliasInfoType:
        if isinstance(parsed_type, ModifiedType):
            parsed_type.base_type = AliasInfoType(parsed_type.base_type, alias_info)
        else:
            parsed_type = AliasInfoType(parsed_type, alias_info)
        return parsed_type

    def parse_type(self) -> Type:
        parsed_type, alias_info = self._parse_type_and_alias()
        if not alias_info:
            return parsed_type
        return self._create_alias_info_type(parsed_type, alias_info)

    def _parse_type_and_alias(self) -> Tuple[Type, AliasInfo]:
        parsed_type: Type = None
        alias_info: AliasInfo = None

        if self._peek_token(TokenKind.MUL):
            return (KWArgsSentinelType(self._read_token()), None)

        if self._peek_token(TokenKind.OPEN_PAREN):

            def parse_tuple_element():
                element_type, element_alias_info = self._parse_type_and_alias()
                if element_alias_info:
                    element_type = self._create_alias_info_type(element_type, element_alias_info)

                return TupleMemberType(
                    element_type, self._read_token() if self._peek_token(TokenKind.IDENTIFIER) else None
                )

            parsed_type = TupleType(
                self._parse_list(TokenKind.OPEN_PAREN, TokenKind.COMMA, TokenKind.CLOSE_PAREN, parse_tuple_element)
            )
        elif self._peek_token(TokenKind.IDENTIFIER, "Tensor"):
            parsed_type = TensorType(self._read_token())
            alias_info = self._parse_torch_alias_info()
        else:
            parsed_type = self._parse_torch_base_type()
            alias_info = self._parse_torch_alias_info()

        while True:
            if self._peek_token(TokenKind.OPEN_BRACKET):
                parsed_type = ArrayType(
                    parsed_type,
                    self._read_token(),
                    self._read_token() if self._peek_token(TokenKind.NUMBER) else None,
                    self._expect_token(TokenKind.CLOSE_BRACKET),
                )
            elif self._peek_token(TokenKind.QUESTION_MARK):
                parsed_type = OptionalType(parsed_type, self._read_token())
            else:
                return (parsed_type, alias_info)

    def _parse_torch_base_type(self) -> Type:
        base_type_parsers = {
            "int": IntType,
            "float": FloatType,
            "bool": BoolType,
            "str": StrType,
            "Scalar": ScalarType,
            "ScalarType": ScalarTypeType,
            "Dimname": DimnameType,
            "Layout": LayoutType,
            "Device": DeviceType,
            "Generator": GeneratorType,
            "MemoryFormat": MemoryFormatType,
            "QScheme": QSchemeType,
            "Storage": StorageType,
            "ConstQuantizerPtr": ConstQuantizerPtrType,
            "Stream": StreamType,
            "SymInt": SymIntType,
        }
        identifier = self._expect_token(TokenKind.IDENTIFIER)
        base_type_parser = base_type_parsers.get(identifier.value)
        if not base_type_parser:
            raise ExpectedSyntaxError("|".join(base_type_parsers.keys()), identifier)
        base_type = base_type_parser(identifier)
        return base_type

    def _parse_torch_alias_info(self) -> AliasInfo:
        alias_info = AliasInfo()

        def parse_set(alias_set: List[str]):
            while True:
                if self._peek_token(TokenKind.MUL):
                    alias_info.tokens.append(self._read_token())
                    alias_set.append("*")
                elif "*" not in alias_set:
                    identifier = self._expect_token(TokenKind.IDENTIFIER)
                    alias_info.tokens.append(identifier)
                    alias_set.append(identifier.value)
                else:
                    raise ExpectedSyntaxError("alias wildcard * or alias identifier", self._peek_token())

                if self._peek_token(TokenKind.OR):
                    alias_info.tokens.append(self._read_token())
                else:
                    return

        if self._peek_token(TokenKind.OPEN_PAREN):
            alias_info.tokens.append(self._read_token())

            parse_set(alias_info.before_set)

            if self._peek_token(TokenKind.EXCLAIMATION_MARK):
                alias_info.tokens.append(self._read_token())
                alias_info.is_writable = True

            if self._peek_token(TokenKind.ARROW):
                alias_info.tokens.append(self._read_token())
                parse_set(alias_info.after_set)
            else:
                # no '->' so assume before and after are identical
                alias_info.after_set = alias_info.before_set

            alias_info.tokens.append(self._expect_token(TokenKind.CLOSE_PAREN))
        elif self._peek_token(TokenKind.EXCLAIMATION_MARK):
            alias_info.is_writable = True
            alias_info.before_set.append(str(self._next_anonymous_alias_id))
            self._next_anonymous_alias_id += 1
        else:
            return None

        return alias_info


def cpp_create_from_file(path: str) -> CPPParser:
    return CPPParser(FileReader(path))


def cpp_create_from_string(buffer: str) -> CPPParser:
    return CPPParser(StringReader(buffer))


def torch_create_from_file(path: str) -> TorchParser:
    return TorchParser(FileReader(path))


def torch_create_from_string(buffer: str) -> TorchParser:
    return TorchParser(StringReader(buffer))
