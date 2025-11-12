// Forked from
// https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?path=/src/perceptiveshell_private/perceptiveshell_private/include/perceptiveshell_private/arg_parser.h&version=GBmain&_a=contents
// at commit cb10f87fd4b130682f5a65dfa42aa34deeb40fa6.

#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::wstring TrimLeft(const std::wstring& str, const char charToTrim = ' ')
{
    std::wstring trimmed(str);
    trimmed.erase(
        trimmed.begin(),
        std::find_if(trimmed.begin(), trimmed.end(), [charToTrim](const wchar_t c) { return c != charToTrim; }));
    return trimmed;
}

typedef struct _Arg
{
    std::wstring longName;
    std::wstring shortName;
    bool isSwitch;
    bool isRequired;
    std::wstring helpText;
    std::wstring value;

    bool Is(const std::wstring& name) const
    {
        return (name == longName) || (name == shortName) || (name == TrimLeft(longName, '-'));
    }
} Arg;

typedef std::vector<Arg>::iterator ArgIter;

/// <summary>
///     Simplistic header-only argument parser.
///     Expects only arguments with a single or no values (switches).
///     Stores arguments as strings, parses them on demand.
/// </summary>
class ArgParser
{
public:
    bool AddArgument(
        const std::wstring& longName,
        const std::wstring& shortName = L"",
        const std::wstring& helpText = L"",
        bool isRequired = false,
        bool isSwitch = false,
        const std::wstring& defaultValue = L"")
    {
        if (longName.empty())
        {
            std::wcout << "Long name cannot be empty." << std::endl;
            return false;
        }

        // Check if duplicate long name.
        ArgIter argIter = FindArg(longName);
        if (argIter != m_args.end())
        {
            std::wcout << "Argument already specified: " << argIter->longName << " " << argIter->shortName << std::endl;
            return false;
        }

        // Check if duplicate short name.
        argIter = FindArg(shortName);
        if (argIter != m_args.end())
        {
            std::wcout << "Argument already specified: " << argIter->longName << " " << argIter->shortName << std::endl;
            return false;
        }

        // Switches cannot be required.
        if (isSwitch)
        {
            if (isRequired)
            {
                std::wcout << longName << " specified both as switch and required." << std::endl;
                return false;
            }
            if (!defaultValue.empty())
            {
                std::wcout << "Switches cannot have a default value: " << longName << "=" << defaultValue << std::endl;
                std::wcout << "Please leave it blank." << std::endl;
                return false;
            }
        }

        m_args.push_back(Arg{longName, shortName, isSwitch, isRequired, helpText, defaultValue});
        return true;
    }

    bool DeleteArgument(const std::wstring& longName)
    {
        if (longName.empty())
        {
            std::wcout << "Long name cannot be empty." << std::endl;
            return false;
        }

        // Check if duplicate long name.
        ArgIter argIter = FindArg(longName);
        if (argIter == m_args.end())
        {
            std::wcout << "Argument doesn't exist: " << argIter->longName << " " << argIter->shortName << std::endl;
            return false;
        }

        m_args.erase(argIter);
        return true;
    }

    bool ParseArgs(int argc, const wchar_t** argv)
    {
        m_parsed_yet = true;
        m_appName = argv[0];

        for (int argIdx = 1; argIdx < argc; ++argIdx)
        {
            // Get long or short name.
            std::wstring argName(argv[argIdx]);
            if (argName[0] != '-')
            {
                std::wcout << L"Unexpected argument: " << argName << std::endl;
                std::wcout << L"It should be either in the form of --longName or -s" << std::endl;
                return false;
            }

            // Find the argument specification.
            ArgIter argIter = FindArg(argName);
            if (argIter == m_args.end())
            {
                std::wcout << argName << " not supported." << std::endl;
                return false;
            }

            // If argument is a switch, set it to true and don't look further.
            if (argIter->isSwitch)
            {
                argIter->value = L"true"; // Not important, because any non-empty value will evaluate to true.
                continue;
            }

            // Read argument's value.
            ++argIdx;
            if (argIdx >= argc)
            {
                std::wcout << "Not enough arguments." << std::endl;
                return false;
            }

            std::wstring value(argv[argIdx]);
            if (FindArg(value) != m_args.end())
            {
                // The next item actually matches a known argument though it should be the parameter
                // of the current argument.
                std::wcout << "Missing value for " << argName << std::endl;
                return false;
            }

            argIter->value = value;
        }

        bool areArgsValid = ValidateArgs();
        return areArgsValid;
    }

    bool ParseArgs(const std::vector<std::wstring>& args)
    {
        std::vector<const wchar_t*> argPtrs;
        std::transform(
            args.begin(), args.end(), std::back_inserter(argPtrs), [](const std::wstring& arg) { return arg.data(); });
        return ParseArgs(static_cast<int>(argPtrs.size()), argPtrs.data());
    }

    std::wstring GetHelpText()
    {
        std::wstringstream helpText;
        if (m_args.empty())
        {
            std::wcout << "Please add some arguments first, with AddArgument()." << std::endl;
            return L"";
        }

        for (Arg const& arg : m_args)
        {
            // Enclose parameter in [] if optional.
            if (!arg.isRequired)
            {
                helpText << "[";
            }

            // --longName (-s) LONGNAME
            helpText << arg.longName;
            if (!arg.shortName.empty())
            {
                helpText << " (" << arg.shortName << ")";
            }
            if (!arg.isSwitch)
            {
                // Trim the leading dashes:
                // --longName -> longName
                std::wstring valueName = TrimLeft(arg.longName, '-');

                // Convert to uppercase:
                // longName -> LONGNAME
                std::transform(
                    valueName.begin(),
                    valueName.end(),
                    valueName.begin(),
                    [](const wchar_t& c) { return static_cast<wchar_t>(std::toupper(static_cast<int>(c))); });

                helpText << " " << valueName;
            }

            // Enclose parameter in [] if optional.
            if (!arg.isRequired)
            {
                helpText << "]";
            }

            helpText << std::endl;

            // Help text, indented.
            const std::wstring c_indentation(L"    ");
            helpText << c_indentation << arg.helpText << std::endl;

            // Default value, if optional argument.
            if (!arg.isRequired && !arg.isSwitch)
            {
                helpText << c_indentation << "Default: " << arg.value << std::endl;
            }
        }

        return helpText.str();
    }

    bool IsArg(const std::wstring& name)
    {
        return FindArg(name) != m_args.end();
    }

    std::wstring Get(const std::wstring& name)
    {
        ArgIter argIter = FindArgThrow(name);
        return argIter->value;
    }

    bool GetBool(const std::wstring& name)
    {
        ArgIter argIter = FindArgThrow(name);
        return !argIter->value.empty();
    }

    int GetInt(const std::wstring& name)
    {
        ArgIter argIter = FindArgThrow(name);
        return std::stoi(argIter->value);
    }

    float GetFloat(const std::wstring& name)
    {
        ArgIter argIter = FindArgThrow(name);
        return std::stof(argIter->value);
    }

private:
    ArgIter FindArg(const std::wstring& name)
    {
        auto argIter = std::find_if(m_args.begin(), m_args.end(), [&name](const Arg& arg) { return arg.Is(name); });

        return argIter;
    }

    const ArgIter FindArgThrow(const std::wstring& name)
    {
        if (!m_parsed_yet)
        {
            throw std::runtime_error("Looking for an argument value before it is set.");
        }
        const ArgIter argIter = FindArg(name);
        if (argIter == m_args.end())
        {
            throw std::invalid_argument("Argument not found.");
        }
        return argIter;
    }

    bool ValidateArgs()
    {
        bool success = true;
        for (ArgIter argIter = m_args.begin(); argIter != m_args.end(); ++argIter)
        {
            if (argIter->isRequired && argIter->value.empty())
            {
                std::wcout << "Required argument: " << argIter->longName << std::endl;
                success = false;
            }
        }
        return success;
    }

private:
    bool m_parsed_yet{false};
    std::vector<Arg> m_args;
    std::wstring m_appName;
};
