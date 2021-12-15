// class SH: table of spherical harmonics coefficients,
// used to approximate a spherical function

#pragma once

const float PI = acos(-1);


struct TensorEntry
{
    short a,b,c;
    float val;
};


// trivially copyable; trivially destructible
class SH
{
	static int SHIndex(int l, int m) {
		return l*l+l+m;
	}
public:
    int n; // SH order
    std::vector<float> a; // coefficients

	float& at(int l, int m) {
		return a[SHIndex(l,m)];
	}
    float const& at(int l, int m) const {
        return a[SHIndex(l,m)];
    }

	SH(int n): n(n), a(n*n, 0.f) {}
    float magnitude() const;
};

SH operator-(const SH& a, const SH& b);

// ============== implementation starts ==============

float SH::magnitude() const
{
    float t = 0;
    for (int i=0; i<n*n; ++i)
        t += a[i] * a[i];
    return std::sqrt(t);
}

SH operator-(const SH& a, const SH& b)
{
    if (a.n!=b.n) throw "order mismatch";
    int n = a.n;
    SH c(n);
    for (int l=0; l<n; ++l)
        for (int m=-l; m<=l; ++m)
            c.at(l,m) = a.at(l,m) - b.at(l,m);
    return c;
}

