using System;
using System.ComponentModel;

namespace Microsoft.ML.OnnxRuntime
{
	/// <summary>
	/// Preserve attribute to prevent the MonoTouch linker from linking the target.
	/// <see cref="https://docs.microsoft.com/en-us/dotnet/api/foundation.preserveattribute?view=xamarin-ios-sdk-12"/>
	/// </summary>
	[AttributeUsage(AttributeTargets.All)]
	[EditorBrowsable(EditorBrowsableState.Never)]
	public sealed class PreserveAttribute : Attribute
	{
		/// <summary>
		/// Ensures that all members of this type are preserved.
		/// </summary>
		public bool AllMembers;

		/// <summary>
		/// Flags the method as a method to preserve during linking if the container class is pulled in.
		/// </summary>
		public bool Conditional;

		/// <summary>
		/// Instruct the MonoTouch linker to preserve the decorated code
		/// </summary>
		/// <param name="allMembers">Ensures that all members of this type are preserved.</param>
		/// <param name="conditional">Flags the method as a method to preserve during linking if the container class is pulled in.</param>
		public PreserveAttribute(bool allMembers, bool conditional)
		{
			AllMembers = allMembers;
			Conditional = conditional;
		}

		public PreserveAttribute()
		{
		}
	}
}