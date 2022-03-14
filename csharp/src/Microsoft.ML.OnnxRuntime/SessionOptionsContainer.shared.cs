// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Helper to allow the creation/addition of session options based on pre-defined named entries.
    /// </summary>
    public static class SessionOptionsContainer
    {
        static Lazy<Action<SessionOptions>> _defaultHandler;

        static readonly Dictionary<string, Lazy<Action<SessionOptions>>> _configurationHandlers =
            new Dictionary<string, Lazy<Action<SessionOptions>>>();

        static Lazy<Action<SessionOptions>> DefaultHandler =>
            _defaultHandler != null
                ? _defaultHandler
                : (_defaultHandler = new Lazy<Action<SessionOptions>>(() => (options) => { /* use as is */ }));

        /// <summary>
        /// Register the default handler. This is used when a configuration name is not provided.
        /// </summary>
        /// <param name="defaultHandler">Handler that applies the default settings to a SessionOptions instance.
        /// </param>
        public static void Register(Action<SessionOptions> defaultHandler) => _defaultHandler =
            new Lazy<Action<SessionOptions>>(() => defaultHandler);

        /// <summary>
        /// Register a named handler.
        /// </summary>
        /// <param name="configuration">Configuration name.</param>
        /// <param name="handler">
        /// Handler that applies the settings for the configuration to a SessionOptions instance.
        /// </param>
        public static void Register(string configuration, Action<SessionOptions> handler) =>
            _configurationHandlers[configuration] = new Lazy<Action<SessionOptions>>(() => handler);

        /// <summary>
        /// Create a SessionOptions instance with configuration applied.
        /// </summary>
        /// <param name="configuration">
        /// Configuration to use. 
        /// If not provided, the default set of session options will be applied if useDefaultAsFallback is true.
        /// </param>
        /// <param name="useDefaultAsFallback">
        /// If configuration is not provided or not found, use the default session options.
        /// </param>
        /// <returns>SessionOptions with configuration applied.</returns>
        public static SessionOptions Create(string configuration = null, bool useDefaultAsFallback = true) =>
            new SessionOptions().ApplyConfiguration(configuration, useDefaultAsFallback);

        /// <summary>
        /// Reset by removing all registered handlers.
        /// </summary>
        public static void Reset()
        {
            _defaultHandler = null;
            _configurationHandlers.Clear();
        }

        /// <summary>
        /// Apply a configuration to a SessionOptions instance.
        /// </summary>
        /// <param name="options">SessionOptions to apply configuration to.</param>
        /// <param name="configuration">Configuration to use.</param>
        /// <param name="useDefaultAsFallback">
        /// Use the default configuration if 'configuration' is not provided or not found.
        /// </param>
        /// <returns>Updated SessionOptions instance.</returns>
        public static SessionOptions ApplyConfiguration(this SessionOptions options, string configuration = null,
                                                        bool useDefaultAsFallback = true)
        {
            var handler = Resolve(configuration, useDefaultAsFallback);
            handler(options);

            return options;
        }

        static Action<SessionOptions> Resolve(string configuration = null, bool useDefaultAsFallback = true)
        {
            if (string.IsNullOrWhiteSpace(configuration))
                return DefaultHandler.Value;

            if (_configurationHandlers.TryGetValue(configuration, out var handler))
                return handler.Value;

            if (useDefaultAsFallback)
                return DefaultHandler.Value;

            throw new KeyNotFoundException($"Configuration not found for '{configuration}'");
        }
    }
}
