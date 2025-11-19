// half.hpp - IEEE 754-based half-precision floating point library
// This is a simplified implementation for CUDATrader

#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <iostream>

namespace half_float {

class half {
private:
    uint16_t data_;

public:
    // Constructors
    constexpr half() noexcept : data_(0) {}
    
    // Convert from float
    half(float value) noexcept {
        union {
            float f;
            uint32_t u;
        } conv;
        
        conv.f = value;
        
        // Extract components from float
        uint32_t sign = (conv.u >> 31) & 0x1;
        uint32_t exp = (conv.u >> 23) & 0xFF;
        uint32_t mantissa = conv.u & 0x7FFFFF;
        
        // Handle special cases
        if (exp == 0xFF) { // Infinity or NaN
            exp = 0x1F;
            mantissa = (mantissa ? 0x200 : 0);
        } else if (exp == 0) { // Zero or denormal
            exp = 0;
            mantissa = 0;
        } else { // Normal number
            // Adjust exponent bias (float is 127, half is 15)
            int newExp = exp - 127 + 15;
            
            if (newExp >= 0x1F) { // Overflow, convert to infinity
                exp = 0x1F;
                mantissa = 0;
            } else if (newExp <= 0) { // Underflow, convert to zero
                exp = 0;
                mantissa = 0;
            } else {
                exp = newExp;
                // Shift mantissa to fit in 10 bits
                mantissa = (mantissa >> 13);
            }
        }
        
        // Combine components
        data_ = (sign << 15) | (exp << 10) | mantissa;
    }
    
    // Explicit constructor from integer types
    explicit half(int value) noexcept : half(static_cast<float>(value)) {}
    
    // Constructor from double type (non-explicit for easier conversions)
    half(double value) noexcept : half(static_cast<float>(value)) {}
    
    // Convert to float
    operator float() const noexcept {
        // Extract components
        uint16_t sign = (data_ >> 15) & 0x1;
        uint16_t exp = (data_ >> 10) & 0x1F;
        uint16_t mantissa = data_ & 0x3FF;
        
        union {
            float f;
            uint32_t u;
        } conv;
        
        if (exp == 0) { // Zero or denormal
            conv.u = sign << 31;
        } else if (exp == 0x1F) { // Infinity or NaN
            conv.u = (sign << 31) | (0xFF << 23) | (mantissa ? 0x400000 : 0);
        } else { // Normal number
            // Adjust exponent bias (half is 15, float is 127)
            uint32_t newExp = exp - 15 + 127;
            // Shift mantissa to fit in 23 bits
            uint32_t newMantissa = mantissa << 13;
            
            conv.u = (sign << 31) | (newExp << 23) | newMantissa;
        }
        
        return conv.f;
    }
    
    // Get raw bits
    uint16_t bits() const noexcept {
        return data_;
    }
    
    // Set from raw bits
    void set_bits(uint16_t bits) noexcept {
        data_ = bits;
    }
    
    // Arithmetic operators
    half operator+(const half& rhs) const {
        return half(float(*this) + float(rhs));
    }
    
    half operator-(const half& rhs) const {
        return half(float(*this) - float(rhs));
    }
    
    half operator*(const half& rhs) const {
        return half(float(*this) * float(rhs));
    }
    
    half operator/(const half& rhs) const {
        return half(float(*this) / float(rhs));
    }
    
    // Assignment operators
    half& operator+=(const half& rhs) {
        *this = *this + rhs;
        return *this;
    }
    
    half& operator-=(const half& rhs) {
        *this = *this - rhs;
        return *this;
    }
    
    half& operator*=(const half& rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    half& operator/=(const half& rhs) {
        *this = *this / rhs;
        return *this;
    }
    
    // Comparison operators
    bool operator==(const half& rhs) const {
        return float(*this) == float(rhs);
    }
    
    bool operator!=(const half& rhs) const {
        return float(*this) != float(rhs);
    }
    
    bool operator<(const half& rhs) const {
        return float(*this) < float(rhs);
    }
    
    bool operator>(const half& rhs) const {
        return float(*this) > float(rhs);
    }
    
    bool operator<=(const half& rhs) const {
        return float(*this) <= float(rhs);
    }
    
    bool operator>=(const half& rhs) const {
        return float(*this) >= float(rhs);
    }
    
    // Stream operators
    friend std::ostream& operator<<(std::ostream& os, const half& h) {
        os << float(h);
        return os;
    }
    
    // Helper math functions
    static half abs(const half& h) {
        half result;
        result.data_ = h.data_ & 0x7FFF; // Clear sign bit
        return result;
    }
    
    static half sqrt(const half& h) {
        return half(std::sqrt(float(h)));
    }
};

// Non-member operators for mixed-type operations
// half + double/int
inline half operator+(const half& lhs, double rhs) {
    return half(static_cast<float>(float(lhs) + rhs));
}

inline half operator+(double lhs, const half& rhs) {
    return half(static_cast<float>(lhs + float(rhs)));
}

// half - double/int
inline half operator-(const half& lhs, double rhs) {
    return half(static_cast<float>(float(lhs) - rhs));
}

inline half operator-(double lhs, const half& rhs) {
    return half(static_cast<float>(lhs - float(rhs)));
}

// half * double/int
inline half operator*(const half& lhs, double rhs) {
    return half(static_cast<float>(float(lhs) * rhs));
}

inline half operator*(double lhs, const half& rhs) {
    return half(static_cast<float>(lhs * float(rhs)));
}

// half / double/int
inline half operator/(const half& lhs, double rhs) {
    return half(static_cast<float>(float(lhs) / rhs));
}

inline half operator/(double lhs, const half& rhs) {
    return half(static_cast<float>(lhs / float(rhs)));
}

// Comparison operators
// half < double/int
inline bool operator<(const half& lhs, double rhs) {
    return float(lhs) < rhs;
}

inline bool operator<(double lhs, const half& rhs) {
    return lhs < float(rhs);
}

// half > double/int
inline bool operator>(const half& lhs, double rhs) {
    return float(lhs) > rhs;
}

inline bool operator>(double lhs, const half& rhs) {
    return lhs > float(rhs);
}

// half <= double/int
inline bool operator<=(const half& lhs, double rhs) {
    return float(lhs) <= rhs;
}

inline bool operator<=(double lhs, const half& rhs) {
    return lhs <= float(rhs);
}

// half >= double/int
inline bool operator>=(const half& lhs, double rhs) {
    return float(lhs) >= rhs;
}

inline bool operator>=(double lhs, const half& rhs) {
    return lhs >= float(rhs);
}

// half == double/int
inline bool operator==(const half& lhs, double rhs) {
    return float(lhs) == rhs;
}

inline bool operator==(double lhs, const half& rhs) {
    return lhs == float(rhs);
}

// half != double/int
inline bool operator!=(const half& lhs, double rhs) {
    return float(lhs) != rhs;
}

inline bool operator!=(double lhs, const half& rhs) {
    return lhs != float(rhs);
}

// Math functions
inline half h_sqrt(const half& h) {
    return half::sqrt(h);
}

inline half h_abs(const half& h) {
    return half::abs(h);
}

} // namespace half_float
