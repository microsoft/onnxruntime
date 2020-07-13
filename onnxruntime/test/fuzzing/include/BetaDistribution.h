// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef __BETADISTRIBUTION_H__
#define __BETADISTRIBUTION_H__
#include <random>
#include <map>
#include <chrono>

// Default parameter will produce a shape with alpha = 0.5
// and beta = 0.5.
// By default this distribution creates a standard Beta distribution
//

template< typename result_type,
  typename generator = std::default_random_engine>
class BetaDistribution
{
public:
    // The type used for internal calculations.
    // Should be big enough to avoid overflows
    // internally.
    //
    using calc_type = long double;

    // The type used to store the parameters of
    // the distribution.
    //
    using param_type = std::map<std::string, float>;

    // Create a BetaDistribution with all the necessary parameters
    //
    BetaDistribution(float alpha, float beta, 
        result_type beginRange, result_type endRange)
    : m_alpha(alpha), 
    m_beta(beta), 
    m_beginRange(beginRange),
    m_endRange(endRange)
    { 
        init();
    }

    // Create a BetaDistribution specifying only the range
    // with default parameters alpha = 0.5 and beta = 0.5
    //
    BetaDistribution(result_type beginRange, result_type endRange)
    : m_beginRange(beginRange), 
    m_endRange(endRange)
    {
        init();
    }

    // Create a default Standard BetaDistribution with a shape
    // that defined by alpha = 0.5 and beta = 0.5
    //
    explicit BetaDistribution(){}

    // Get the parameters of the distribution
    //
    param_type param() const
    {
        return param;
    }


    // Get the lowest value that can be generated
    //
    result_type min() const
    {
        return m_beginRange;
    }

    // Get the highest value that can be generated
    //
    result_type max() const
    {
        return m_endRange;
    }

    // Generate a number from beginRange to endRange
    //
    result_type operator()(generator& gen)
    {
        // Find the probability of the each number
        // and return the number with the highest probability
        //
        calc_type highest_probability = 0.0;
        calc_type likely_number = 0;
        for(int i=0; i < sample_size; i++)
        {
            calc_type sample = convert_to_fixed_range(gen);
            calc_type highest_probability_temp = highest_probability;
            highest_probability = std::max({ highest_probability_temp, distribution(sample)});

            // A new sample number with a higher probabilty has been found
            //
            if (highest_probability > highest_probability_temp)
            {
                likely_number = sample;
            }
        }

        return static_cast<result_type>(likely_number);
    }

private:
    // Internal configuration shared across all constructors
    //
    void init()
    {
        if (m_endRange < m_beginRange)
        {
            throw std::runtime_error("endRange Must be greater than begin range");
        }
        param_value["alpha"] = m_alpha;
        param_value["beta"] = m_beta;
    }

    // A constant value used for internal computation.
    //
    constexpr inline double sqrtpi()
    { 
        return std::sqrt( std::atan(1)*4 ); 
    }

    // Calculates the value of x in a gamma distribution
    //
    template<typename gamma_input_type>
    double calculate_gamma(gamma_input_type x)
    {
        static_assert(std::is_arithmetic<gamma_input_type>(), 
                        "Input to calculate_gamma must be arithmetic");
        if (x < 0)
        {
            throw std::invalid_argument("No implementation for gamma less than 0");
        }
        if (std::is_floating_point<gamma_input_type>() && x <= 0.5 && x> 0.49 )
        {
            return sqrtpi();
        }
        else if (std::is_floating_point<gamma_input_type>() && x <= 1.00 && x >= 0.99)
        {
            return 1;
        }
        else if ( std::is_integral<gamma_input_type>() )
        {
            // Calculate factorial
            //
            gamma_input_type result = 1;
            for(gamma_input_type n = x - 1; n >= 0; n--)
            {
            result = n == 0 ? result*1 : result *n;
            }
            
            return static_cast<double>(result);
        }
        else
        {
            throw std::exception("Non special gamma values not yet Implemeted");
        }
    }

    // Generate the probabilty of having this number
    //
    inline calc_type distribution(calc_type randVar)
    {
        if (randVar > max() || randVar < min())
        {
            return 0;
        }
        calc_type range {static_cast<calc_type>(max()) - static_cast<calc_type>(min())};
        calc_type term {1.0/range};

        calc_type gammaTerm {calculate_gamma(m_alpha + m_beta)};
        calc_type gammaTerm1 {calculate_gamma(m_alpha)};
        calc_type gammaTerm2 {calculate_gamma(m_beta)};
        calc_type term1 {gammaTerm/(gammaTerm1 * gammaTerm2)};

        calc_type term2 { pow( (randVar - min()) / range, m_alpha - 1)};
        calc_type term3 { pow( (max() - randVar) / range, m_beta - 1) };

        return {term * term1 * term2 * term3};
    }

    // Used to convert the number that generator produces
    // to the range specified.
    // For example, the default BetaDistribution
    // generates number between 0..1. Hence this function
    // will convert 0..N produced by the generator to 0..1.  
    //
    calc_type convert_to_fixed_range(generator& gen)
    {
        // Find a number in the generator space
        //
        calc_type x{ static_cast<calc_type>(gen())};

        // Convert the number to the range [beginRange, endRange]
        //
        calc_type range { std::numeric_limits<generator::result_type>::max() 
                                - std::numeric_limits<generator::result_type>::lowest()};

        calc_type delta {x - std::numeric_limits<generator::result_type>::lowest()};                    
        calc_type ratio {delta/range};

        calc_type new_range { static_cast<calc_type>(max()) - static_cast<calc_type>(min())};
        if (new_range <= 0 || new_range >= std::numeric_limits<calc_type>::infinity())
        {
        throw std::runtime_error(
            "Overflow error: The range of the Beta distribution is to big to fit into the result_type.\n"\
            "Consider using the standard and then scaling to the desired range.");
        }

        calc_type res {(ratio * new_range) + min()};
        return {res};
    }

private:
  static constexpr int sample_size = 2;
  float m_alpha = 0.5;
  float m_beta = 0.5;
  result_type m_beginRange = 0;
  result_type m_endRange = 1;
  param_type param_value = 
  {
    std::pair<std::string, float>{"alpha", m_alpha}, 
    std::pair<std::string, float>{"beta", m_beta}
  };
};

// Test to visualize the distribution
//
void unittestBetaDistribution();

// Test to generate Random data
// and verify its distribution.
//
void unittestGenerateRandomData();

// type - Used to determine the size of the data
// numElementsToGenerate - Number of elements to generate
//
template<typename ONNX_ELEMENT_VALUE_TYPE>
std::vector<ONNX_ELEMENT_VALUE_TYPE>
GenerateRandomData(ONNX_ELEMENT_VALUE_TYPE initialValue, size_t numElementsToGenerate, size_t seed)
{
  // Store the generated data in this vector
  //
  std::vector<ONNX_ELEMENT_VALUE_TYPE> randomDataBucket(numElementsToGenerate, initialValue);

  // The Beta distribution is likely to returns values
  // at the extremes of a finite range. For examples for
  // a float, most values generated will be around numeric_limit<float>::min
  // and the numeric_limit<float>::max. To avoid problems of overflow
  // use the standard to generate number in the close to 0 and close to 1.
  //
  std::default_random_engine generator(static_cast<unsigned int>(seed));
  BetaDistribution<ONNX_ELEMENT_VALUE_TYPE> standardBetaDistribution
    { std::numeric_limits<ONNX_ELEMENT_VALUE_TYPE>::min(), 
      std::numeric_limits<ONNX_ELEMENT_VALUE_TYPE>::max()
    };

  // Generate the data in the vector
  //
  for(auto& data: randomDataBucket)
  {
    data = standardBetaDistribution(generator);
  }

  return randomDataBucket;
}
#endif