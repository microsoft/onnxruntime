//
// gsl-lite-vc6 is based on GSL: Guidelines Support Library,
// For more information see https://github.com/martinmoene/gsl-lite
//
// Copyright (c) 2015 Martin Moene
// Copyright (c) 2015 Microsoft Corporation. All rights reserved.
//
// This code is licensed under the MIT License (MIT).
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#ifndef GSL_GSL_LITE_H_INCLUDED
#define GSL_GSL_LITE_H_INCLUDED

#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define  gsl_lite_VERSION "0.0.0"

// Configuration:

#ifndef  gsl_FEATURE_IMPLICIT_MACRO
# define gsl_FEATURE_IMPLICIT_MACRO  1
#endif

#ifndef  gsl_FEATURE_OWNER_MACRO
# define gsl_FEATURE_OWNER_MACRO  1
#endif

#ifndef  gsl_FEATURE_SHARED_PTR
# define gsl_FEATURE_SHARED_PTR  0
#endif

#ifndef  gsl_FEATURE_UNIQUE_PTR
# define gsl_FEATURE_UNIQUE_PTR  0
#endif

#ifndef  gsl_CONFIG_THROWS_FOR_TESTING
# define gsl_CONFIG_THROWS_FOR_TESTING  0
#endif

#ifndef  gsl_CONFIG_CONFIRMS_COMPILATION_ERRORS
# define gsl_CONFIG_CONFIRMS_COMPILATION_ERRORS  0
#endif

#ifndef  gsl_CONFIG_SHARED_PTR_INCLUDE
# define gsl_CONFIG_SHARED_PTR_INCLUDE  <boost/shared_ptr.hpp>
#endif

#ifndef  gsl_CONFIG_UNIQUE_PTR_INCLUDE
# define gsl_CONFIG_UNIQUE_PTR_INCLUDE  <boost/unique_ptr.hpp>
#endif

#ifndef  gsl_CONFIG_SHARED_PTR_DECL
# define gsl_CONFIG_SHARED_PTR_DECL  boost::shared_ptr
#endif

#ifndef  gsl_CONFIG_UNIQUE_PTR_DECL
# define gsl_CONFIG_UNIQUE_PTR_DECL  boost::unique_ptr
#endif

// Compiler detection:

#if defined(_MSC_VER ) && !defined(__clang__)
# define gsl_COMPILER_MSVC_VER      (_MSC_VER )
# define gsl_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900 ) ) )
#else
# define gsl_COMPILER_MSVC_VER       0
# define gsl_COMPILER_MSVC_VERSION   0
# define gsl_COMPILER_NON_MSVC       1
#endif

#if gsl_COMPILER_MSVC_VERSION != 60
# error GSL Lite: this header is for Visual C++ 6
#endif

// half-open range [lo..hi):
#define gsl_BETWEEN( v, lo, hi ) ( (lo) <= (v) && (v) < (hi) )

// Presence of C++ language features:

// C++ feature usage:

#if gsl_FEATURE_IMPLICIT_MACRO
# define implicit
#endif

#define gsl_DIMENSION_OF( a ) ( sizeof(a) / sizeof(0[a]) )

#if gsl_FEATURE_SHARED_PTR
# include gsl_CONFIG_SHARED_PTR_INCLUDE
#endif

#if gsl_FEATURE_UNIQUE_PTR
# include gsl_CONFIG_UNIQUE_PTR_INCLUDE
#endif

namespace gsl {

//
// GSL.owner: ownership pointers
//
// ToDo:
#if gsl_FEATURE_SHARED_PTR
  using gsl_CONFIG_SHARED_PTR_DECL;
#endif
#if gsl_FEATURE_UNIQUE_PTR
  using gsl_CONFIG_UNIQUE_PTR_DECL;
#endif

template< class T > struct owner { typedef T type; };

#define gsl_HAVE_OWNER_TEMPLATE  0

#if gsl_FEATURE_OWNER_MACRO
# define Owner(t)  ::gsl::owner<t>::type
#endif

//
// GSL.assert: assertions
//
#define Expects(x)  ::gsl::fail_fast_assert((x))
#define Ensures(x)  ::gsl::fail_fast_assert((x))

#if gsl_CONFIG_THROWS_FOR_TESTING

struct fail_fast : public std::runtime_error
{
    fail_fast()
    : std::runtime_error( "GSL assertion" ) {}

    explicit fail_fast( char const * const message )
    : std::runtime_error( message ) {}
};

inline void fail_fast_assert( bool cond )
{
    if ( !cond )
        throw fail_fast();
}

inline void fail_fast_assert( bool cond, char const * const message )
{
    if ( !cond )
        throw fail_fast( message );
}

#else // gsl_CONFIG_THROWS_FOR_TESTING

inline void fail_fast_assert( bool cond )
{
    if ( !cond )
        terminate();
}

inline void fail_fast_assert( bool cond, char const * const )
{
    if ( !cond )
        terminate();
}

#endif // gsl_CONFIG_THROWS_FOR_TESTING

//
// GSL.util: utilities
//

class final_action
{
public:
    typedef void (*Action)();

    final_action( Action action )
    : action_( action ) {}

    ~final_action()
    {
        action_();
    }

private:
    Action action_;
};

template< class Fn >
final_action finally( Fn const & f )
{
    return final_action(( f ));
}

template< class T, class U >
T narrow_cast( U u )
{
    return static_cast<T>( u );
}

struct narrowing_error : public std::exception {};

template< class T, class U >
T narrow( U u )
{
    T t = narrow_cast<T>( u );

    if ( static_cast<U>( t ) != u )
    {
        throw narrowing_error();
    }
    return t;
}

//
// GSL.views: views
//

//
// at() - Bounds-checked way of accessing static arrays, std::array, std::vector.
//

namespace detail {

struct precedence_0 {};
struct precedence_1 : precedence_0 {};
struct order_precedence : precedence_1 {};

template< class Array, class T >
T & at( Array & arr, size_t index, T*, precedence_0 const & )
{
    Expects( index < gsl_DIMENSION_OF( arr ) );
    return arr[index];
}

} // namespace detail

// Create an at( container ) function:

# define gsl_MK_AT( Cont ) \
    namespace gsl { namespace detail { \
    template< class T > \
    inline T & at( Cont<T> & cont, size_t index, T*, precedence_1 const & ) \
    { \
        Expects( index < cont.size() ); \
        return cont[index]; \
    } }}

template< class Cont >
int & at( Cont & cont, size_t index )
{
    return detail::at( cont, index, &cont[0], detail::order_precedence() );
}

//
// not_null<> - Wrap any indirection and enforce non-null.
//
template<class T>
class not_null
{
public:
    not_null(             T t         ) : ptr_ ( t ){ Expects( ptr_ != NULL ); }
    not_null & operator=( T const & t ) { ptr_ = t ;  Expects( ptr_ != NULL ); return *this; }

    not_null(             not_null const & other ) : ptr_ ( other.ptr_  ) {}
    not_null & operator=( not_null const & other ) { ptr_ = other.ptr_; }

    // VC6 accepts this anyway:
    // template< typename U > not_null( not_null<U> const & other );
    // template< typename U > not_null & operator=( not_null<U> const & other ) ;

private:
    // Prevent compilation when initialized with a literal 0:
    not_null(             int );
    not_null & operator=( int );

public:
    T get() const
    {
        return ptr_;
    }

         operator T() const { return get(); }
    T    operator->() const { return get(); }

    bool operator==(T const & rhs) const { return    ptr_ == rhs; }
    bool operator!=(T const & rhs) const { return !(*this == rhs); }

private:
    T ptr_;

    not_null & operator++();
    not_null & operator--();
    not_null   operator++( int );
    not_null   operator--( int );
    not_null & operator+ ( size_t );
    not_null & operator+=( size_t );
    not_null & operator- ( size_t );
    not_null & operator-=( size_t );
};

//
// Byte-specific type.
//
typedef unsigned char byte;

//
// span<> - A 1D view of contiguous T's, replace (*,len).
//
template< class T >
class span
{
public:
    typedef size_t size_type;

    typedef T value_type;
    typedef T & reference;
    typedef T * pointer;
    typedef T const * const_pointer;

    typedef pointer       iterator;
    typedef const_pointer const_iterator;

    typedef std::reverse_iterator< iterator, T >             reverse_iterator;
    typedef std::reverse_iterator< const_iterator, const T > const_reverse_iterator;

    // Todo:
    // typedef typename std::iterator_traits< iterator >::difference_type difference_type;

    span()
        : begin_( NULL )
        , end_  ( NULL )
    {
        Expects( size() == 0 );
    }

    span( pointer begin, pointer end )
        : begin_( begin )
        , end_  ( end )
    {
        Expects( begin <= end );
    }

    span( pointer data, size_type size )
        : begin_( data )
        , end_  ( data + size )
    {
        Expects( size == 0 || ( size > 0 && data != NULL ) );
    }

private:
    struct precedence_0 {};
    struct precedence_1 : precedence_0 {};
    struct precedence_2 : precedence_1 {};
    struct order_precedence : precedence_1 {};

    template< class Array, class U >
    span create( Array & arr, U*, precedence_0 const & ) const
    {
        return span( arr, gsl_DIMENSION_OF( arr ) );
    }

    span create( std::vector<T> & cont, T*, precedence_1 const & ) const
    {
        return span( &cont[0], cont.size() );
    }

public:
    template< class Cont >
    span( Cont & cont )
    {
        *this = create( cont, &cont[0], order_precedence() );
    }

#if 0
    // =default constructor
    span( span const & other )
        : begin_( other.begin() )
        , end_  ( other.end() )
    {}
#endif

    span & operator=( span const & other )
    {
        // VC6 balks at copy-swap implementation (here),
        // so we do it the simple way:
        begin_ = other.begin_;
        end_   = other.end_;
        return *this;
    }

#if 0
    // Converting from other span ?
    template< typename U > operator=();
#endif

    iterator begin() const
    {
        return iterator( begin_ );
    }

    iterator end() const
    {
        return iterator( end_ );
    }

    const_iterator cbegin() const
    {
        return const_iterator( begin() );
    }

    const_iterator cend() const
    {
        return const_iterator( end() );
    }

    reverse_iterator rbegin() const
    {
        return reverse_iterator( end() );
    }

    reverse_iterator rend() const
    {
        return reverse_iterator( begin() );
    }

    const_reverse_iterator crbegin() const
    {
        return const_reverse_iterator( cend() );
    }

    const_reverse_iterator crend() const
    {
        return const_reverse_iterator( cbegin() );
    }

    operator bool () const
    {
        return begin_ != NULL;
    }

    reference operator[]( size_type index )
    {
        return at( index );
    }

    bool operator==( span const & other ) const
    {
        return  size() == other.size()
            && (begin_ == other.begin_ || std::equal( this->begin(), this->end(), other.begin() ) );
    }

    bool operator!=( span const & other ) const
    {
        return !( *this == other );
    }

    bool operator< ( span const & other ) const
    {
        return std::lexicographical_compare( this->begin(), this->end(), other.begin(), other.end() );
    }

    bool operator<=( span const & other ) const
    {
        return !( other < *this );
    }

    bool operator> ( span const & other ) const
    {
        return ( other < *this );
    }

    bool operator>=( span const & other ) const
    {
        return !( *this < other );
    }

    reference at( size_type index )
    {
        Expects( index >= 0 && index < size());
        return begin_[ index ];
    }

    pointer data() const
    {
        return begin_;
    }

    bool empty() const
    {
        return size() == 0;
    }

    size_type size() const
    {
        return std::distance( begin_, end_ );
    }

    size_type length() const
    {
        return size();
    }

    size_type used_length() const
    {
        return length();
    }

    size_type bytes() const
    {
        return sizeof( value_type ) * size();
    }

    size_type used_bytes() const
    {
        return bytes();
    }

    void swap( span & other )
    {
        using std::swap;
        swap( begin_, other.begin_ );
        swap( end_  , other.end_   );
    }

    span< const byte > as_bytes() const
    {
        return span< const byte >( reinterpret_cast<const byte *>( data() ), bytes() );
    }

    span< byte > as_writeable_bytes() const
    {
        return span< byte >( reinterpret_cast<byte *>( data() ), bytes() );
    }

    template< class U >
    struct mk
    {
        static span<U> view( U * data, size_type size )
        {
            return span<U>( data, size );
        }
    };

    template< typename U >
    span< U > as_span( U u = U() ) const
    {
        Expects( ( this->bytes() % sizeof(U) ) == 0 );
        return mk<U>::view( reinterpret_cast<U *>( this->data() ), this->bytes() / sizeof( U ) );
    }

private:
    pointer begin_;
    pointer end_;
};

// span creator functions (see ctors)

template< typename T>
span< const byte > as_bytes( span<T> spn )
{
    return span< const byte >( reinterpret_cast<const byte *>( spn.data() ), spn.bytes() );
}

template< typename T>
span< byte > as_writeable_bytes( span<T> spn )
{
    return span< byte >( reinterpret_cast<byte *>( spn.data() ), spn.bytes() );
}

template< typename T >
span<T> as_span( T * begin, T * end )
{
    return span<T>( begin, end );
}

template< typename T >
span<T> as_span( T * begin, size_t size )
{
    return span<T>( begin, size );
}

namespace detail {

template< class T >
struct mk
{
    static span<T> view( std::vector<T> & cont )
    {
        return span<T>( cont );
    }
};
}

template< class T >
span<T> as_span( std::vector<T> & cont )
{
    return detail::mk<T>::view( cont );
}

//
// String types:
//

typedef char * zstring;
typedef wchar_t * zwstring;
typedef const char * czstring;
typedef const wchar_t * cwzstring;

typedef span< char > string_span;
typedef span< wchar_t > wstring_span;
typedef span< const char > cstring_span;
typedef span< const wchar_t > cwstring_span;

// to_string() allow (explicit) conversions from string_span to string

inline std::string to_string( string_span const & view )
{
    return std::string( view.data(), view.length() );
}

inline std::string to_string( cstring_span const & view )
{
    return std::string( view.data(), view.length() );
}

inline std::wstring to_string( wstring_span const & view )
{
    return std::wstring( view.data(), view.length() );
}

inline std::wstring to_string( cwstring_span const & view )
{
    return std::wstring( view.data(), view.length() );
}

//
// ensure_sentinel()
//
// Provides a way to obtain a span from a contiguous sequence
// that ends with a (non-inclusive) sentinel value.
//
// Will fail-fast if sentinel cannot be found before max elements are examined.
//
namespace detail {

template<class T, class SizeType, const T Sentinel>
struct ensure
{
    static span<T> sentinel( T * seq, SizeType max = (std::numeric_limits<SizeType>::max)() )
    {
        typedef T * pointer;
        typedef typename std::iterator_traits<pointer>::difference_type difference_type;

        pointer cur = seq;

        while ( std::distance( seq, cur ) < static_cast<difference_type>( max ) && *cur != Sentinel )
            ++cur;

        Expects( *cur == Sentinel );

        return span<T>( seq, cur - seq );
    }
};
} // namespace detail

//
// ensure_z - creates a string_span for a czstring or cwzstring.
// Will fail fast if a null-terminator cannot be found before
// the limit of size_type.
//

template< typename T >
span<T> ensure_z( T * sz, size_t max = (std::numeric_limits<size_t>::max)() )
{
    return detail::ensure<T, size_t, 0>::sentinel( sz, max );
}

} // namespace gsl

// at( std::vector ):

gsl_MK_AT( std::vector )

#endif // GSL_GSL_LITE_H_INCLUDED

// end of file
