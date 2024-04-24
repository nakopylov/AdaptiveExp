struct Sastre22 <: ExpApproximant
    label::AbstractString
    c::AbstractVector{Number}
    cost::Rational
end

struct Sastre23 <: ExpApproximant
    label::AbstractString
    c::AbstractVector{Number}
    cost::Rational
end

t15 = Sastre22(
    L"t_{15}^{\left[16\right]}",
    [
        1.0,
        -1.224230230553340 * 10^-1,
        3.484665863364574 * 10^-1,
        -6.331712455883370 * 10^1,
        1.040801735231354 * 10^1,
        -1.491449188999246 * 10^-1,
        -5.792361707073261 * 10^0,
        2.116367017255747 * 10^0,
        2.381070373870987 * 10^-1,
        1.857143141426026 * 10^1,
        2.684264296504340 * 10^-1,
        -6.352311335612147 * 10^-2,
        4.017568440673568 * 10^-1,
        8.712167566050691 * 10^-2,
        2.945531440279683 * 10^-3,
        4.018761610201036 * 10^-4
    ],
    4)

t21 = Sastre23(
    L"t_{21}^{\left[24\right]}",
    [
        1.161658834444880 * 10^-6,
        4.500852739573010 * 10^-6,
        5.374708803114821 * 10^-5,
        2.005403977292901 * 10^-3,
        6.974348269544424 * 10^-2,
        9.418613214806352 * 10^-1,
        2.852960512714315 * 10^-3,
        -7.544837153586671 * 10^-3,
        1.829773504500424,
        3.151382711608315 * 10^-2,
        1.392249143769798 * 10^-1,
        -2.269101241269351 * 10^-3,
        -5.394098846866402 * 10^-2,
        3.112216227982407 * 10^-1,
        9.343851261938047,
        6.865706355662834 * 10^-1,
        3.233370163085380,
        -5.726379787260966,
        -1.413550099309667 * 10^-2,
        -1.638413114712016 * 10^-1
    ], 5)