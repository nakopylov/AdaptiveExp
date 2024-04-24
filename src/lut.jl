function lookup_method(tol, nA, lut)
    m = floor(log10(tol))
    m = clamp(m, minimum(lut[:, 1]), maximum(lut[:, 1]))

    header_offset = 3
    row_idx = findfirst(m -> 10.0^m <= tol, lut[header_offset+1:end, 1])

    if row_idx !== nothing
        i = row_idx + header_offset
        scaling_penalty = 1.1
        tmp = copy(lut[i, :])
        tmp[tmp.<0] .= Inf # because of the LUT format
        scalings = ceil.(log2.(float(nA) ./ tmp))
        scalings[scalings.<0] .= 0 # because of the LUT format
        true_total_costs = lut[1, :] + scalings
        total_costs = lut[1, :] + scalings * scaling_penalty
        idx = argmin(total_costs)
        # method, scalings
        return (lut[3, idx], scalings[idx], true_total_costs[idx])
    else
        return (nothing, 0, 0)
    end
end


function select_method(tol, lut, method)
    row_idx = findall(k -> 10.0^k <= tol, lut[:, 1])
    col_idx = findfirst(m -> m == method, lut[3, :])
    if !isempty(row_idx) && !isempty(col_idx)
        i = minimum(row_idx)
        j = minimum(col_idx)
        if isnothing(method)
            error("Method $(method.label) not found is the $(lut).")
        end
        return (method, lut[i, j])
    else
        error("Method $(method.label) not found is the LUT.")
    end
end


nondiag_lut = reduce(hcat,
    [
        [Inf, Inf, Inf, 0, -1, -2, -3, -3.31133, -4, -5, -6, -7, -7.22472, -8, -9, -1 * 10^(1), -1.1 * 10^(1), -1.2 * 10^(1), -1.3 * 10^(1), -1.4 * 10^(1), -1.5 * 10^(1), -1.54775 * 10^(1), -1.6 * 10^(1)],
        [1, 0, t2, 1.2177, 6.0115 * 10^(-1), 2.2450 * 10^(-1), 7.5281 * 10^(-2), 5.3053 * 10^(-2), 2.4272 * 10^(-2), 7.7235 * 10^(-3), 2.4472 * 10^(-3), 7.7437 * 10^(-4), 5.9789 * 10^(-4), 2.4493 * 10^(-4), 7.7457 * 10^(-5), 2.4495 * 10^(-5), 7.7459 * 10^(-6), 2.4495 * 10^(-6), 7.7460 * 10^(-7), 2.4495 * 10^(-7), 7.7460 * 10^(-8), 4.4703 * 10^(-8), 2.4495 * 10^(-8)],
        [1 + 1 // 3, 0, pr2_1, 2.4587, 1.5679, 8.1950 * 10^(-1), 3.9991 * 10^(-1), 3.1768 * 10^(-1), 1.8970 * 10^(-1), 8.8905 * 10^(-2), 4.1447 * 10^(-2), 1.9277 * 10^(-2), 1.6227 * 10^(-2), 8.9557 * 10^(-3), 4.1586 * 10^(-3), 1.9306 * 10^(-3), 8.9621 * 10^(-4), 4.1600 * 10^(-4), 1.9309 * 10^(-4), 8.9627 * 10^(-5), 4.1602 * 10^(-5), 2.8837 * 10^(-5), 1.9310 * 10^(-5)],
        [2, 0, t4, 1.9495, 1.3745, 8.7041 * 10^(-1), 5.2695 * 10^(-1), 4.4792 * 10^(-1), 3.1019 * 10^(-1), 1.7928 * 10^(-1), 1.0245 * 10^(-1), 5.8147 * 10^(-2), 5.1166 * 10^(-2), 3.2872 * 10^(-2), 1.8540 * 10^(-2), 1.0444 * 10^(-2), 5.8785 * 10^(-3), 3.3075 * 10^(-3), 1.8605 * 10^(-3), 1.0464 * 10^(-3), 5.8849 * 10^(-4), 4.4708 * 10^(-4), 3.3095 * 10^(-4)],
        [2 + 1 // 3, 0, pr4_2, 4.4173, 3.4880, 2.5708, 1.8454, 1.6583, 1.3026, 9.0894 * 10^(-1), 6.2924 * 10^(-1), 4.3331 * 10^(-1), 3.9826 * 10^(-1), 2.9734 * 10^(-1), 2.0356 * 10^(-1), 1.3913 * 10^(-1), 9.5000 * 10^(-2), 6.4820 * 10^(-2), 4.4206 * 10^(-2), 3.0138 * 10^(-2), 2.0542 * 10^(-2), 1.7106 * 10^(-2), 1.4000 * 10^(-2)],
        [3, 0, t8, 3.2356, 2.7121, 2.1751, 1.7192, 1.5945, 1.3454, 1.0441, 8.0450 * 10^(-1), 6.1628 * 10^(-1), 5.8005 * 10^(-1), 4.6986 * 10^(-1), 3.5687 * 10^(-1), 2.7024 * 10^(-1), 2.0417 * 10^(-1), 1.5397 * 10^(-1), 1.1596 * 10^(-1), 8.7238 * 10^(-2), 6.5579 * 10^(-2), 5.7212 * 10^(-2), 4.9268 * 10^(-2)],
        [3 + 1 // 3, 0, pr6_3, 6.2065, 5.2980, 4.3419, 3.5096, 3.2781, 2.8106, 2.2341, 1.7653, 1.3883, 1.3146, 1.0878, 8.5004 * 10^(-1), 6.6279 * 10^(-1), 5.1595 * 10^(-1), 4.0114 * 10^(-1), 3.1157 * 10^(-1), 2.4183 * 10^(-1), 1.8760 * 10^(-1), 1.6615 * 10^(-1), 1.4546 * 10^(-1)],
        [3 + 2 // 3, 0, pr6_4, 7.3458, 6.3643, 5.3042, 4.3663, 4.1026, 3.5656, 2.8935, 2.3364, 1.8793, 1.7888, 1.5071, 1.2059, 9.6331 * 10^(-1), 7.6850 * 10^(-1), 6.1248 * 10^(-1), 4.8778 * 10^(-1), 3.8824 * 10^(-1), 3.0888 * 10^(-1), 2.7689 * 10^(-1), 2.4565 * 10^(-1)],
        [4, 0, t15, 5.70378, 5.17715, 4.60027, 4.06746, 3.91204, 3.58556, 3.15227, 2.76436, 2.41849, 2.34621, 2.11128, 1.83941, 1.59961, 1.38877, 1.20389, 1.04199, 8.99314 * 10^(-1), 7.64696 * 10^(-1), 4.9236 * 10^(-1), 4.63267 * 10^(-1)],
        [4 + 1 // 3, 0, pr8_4, 7.9514, 7.0589, 6.0893, 5.2078, 4.9543, 4.4284, 3.7472, 3.1575, 2.6510, 2.5478, 2.2191, 1.8529, 1.5439, 1.2843, 1.0668, 8.8511 * 10^(-1), 7.3369 * 10^(-1), 6.0771 * 10^(-1), 5.5531 * 10^(-1), 5.0305 * 10^(-1)],
        [4 + 2 // 3, 0, pr8_5, 9.0831, 8.1239, 7.0739, 6.1115, 5.8331, 5.2529, 4.4955, 3.8331, 3.2581, 3.1401, 2.7621, 2.3365, 1.9730, 1.6637, 1.4012, 1.1789, 9.9117 * 10^(-1), 8.3278 * 10^(-1), 7.6620 * 10^(-1), 6.9934 * 10^(-1)],
        [5, 0, t21, 7.33084, 6.84445, 6.29802, 5.77812, 5.62326, 5.29256, 4.84077, 4.42136, 4.03285, 3.94964, 3.67372, 3.34243, 3.03744, 2.75716, 2.49983, 2.26228, 2.03044, 1.54901, 4.54202 * 10^(-1), 4.20914 * 10^(-1)],
        [5, 0, t18, 6.22906, 5.75178, 5.21972, 4.71848, 4.57026, 4.25561, 3.83033, 3.44091, 3.08547, 3.01007, 2.76201, 2.46851, 2.20292, 1.96319, 1.74733, 1.55338, 1.37948, 1.22377, 1.09036, 1.08435],
        [5 + 2 // 3, 0, pr12_8, 1.3585 * 10^(1), 1.2624 * 10^(1), 1.1540 * 10^(1), 1.0507 * 10^(1), 1.0200 * 10^(1), 9.5441, 8.6508, 7.8260, 7.0675, 6.9059, 6.3724, 5.7376, 5.1595, 4.6345, 4.1589, 3.7288, 3.3407, 2.9911, 2.8367, 2.6765],
        [7 + 1 // 3, 0, pr13_13, 1.8103 * 10^(1), 1.7684 * 10^(1), 1.6846 * 10^(1), 1.5696 * 10^(1), 1.5331 * 10^(1), 1.4542 * 10^(1), 1.3448 * 10^(1), 1.2419 * 10^(1), 1.1456 * 10^(1), 1.1249 * 10^(1), 1.0557 * 10^(1), 9.7191, 8.9404, 8.2182, 7.5495, 6.9314, 6.3610, 5.8351, 5.5988, 5.3508]
    ]
)

diag_lut = reduce(hcat,
    [
        [Inf, Inf, Inf, 0.0, -1.0, -2.0, -3.0, -3.31133, -4.0, -5.0, -6.0, -7.0, -7.22472, -8.0, -9.0, -1 * 10^(1), -1.1 * 10^(1), -1.2 * 10^(1), -1.3 * 10^(1), -1.4 * 10^(1), -1.5 * 10^(1), -1.54775 * 10^(1), -1.6 * 10^(1)],
        [2 + 1 // 3, 0, pr2_2, 3.1894, 2.5322, 1.5759, 9.0989 * 10^(-1), 7.6339 * 10^(-1), 5.1596 * 10^(-1), 2.9093 * 10^(-1), 1.6374 * 10^(-1), 9.2104 * 10^(-2), 8.0930 * 10^(-2), 5.1798 * 10^(-2), 2.9129 * 10^(-2), 1.6381 * 10^(-2), 9.2115 * 10^(-3), 5.1800 * 10^(-3), 2.9130 * 10^(-3), 1.6381 * 10^(-3), 9.2116 * 10^(-4), 6.9979 * 10^(-4), 5.1800 * 10^(-4)],
        [3 + 1 // 3, 0, pr3_3, 4.4808, 3.9709, 2.9793, 2.0971, 1.8718, 1.4500, 9.9496 * 10^(-1), 6.8016 * 10^(-1), 4.6413 * 10^(-1), 4.2587 * 10^(-1), 3.1644 * 10^(-1), 2.1566 * 10^(-1), 1.4695 * 10^(-1), 1.0013 * 10^(-1), 6.8218 * 10^(-2), 4.6477 * 10^(-2), 3.1665 * 10^(-2), 2.1573 * 10^(-2), 1.7961 * 10^(-2), 1.4697 * 10^(-2)],
        [3 + 2 // 3, 0, pr4_4, 5.8895, 5.4059, 4.4015, 3.4076, 3.1358, 2.6004, 1.9702, 1.4864, 1.1185, 1.0490, 8.4041 * 10^(-1), 6.3092 * 10^(-1), 4.7343 * 10^(-1), 3.5515 * 10^(-1), 2.6638 * 10^(-1), 1.9978 * 10^(-1), 1.4982 * 10^(-1), 1.1235 * 10^(-1), 9.7928 * 10^(-2), 8.4255 * 10^(-2)],
        [4 + 1 // 3, 0, pr5_5, 7.1999, 6.7906, 5.8144, 4.7596, 4.4590, 3.8495, 3.0946, 2.4777, 1.9783, 1.8802, 1.5766, 1.2550, 9.9825 * 10^(-1), 7.9362 * 10^(-1), 6.3074 * 10^(-1), 5.0119 * 10^(-1), 3.9819 * 10^(-1), 3.1634 * 10^(-1), 2.8342 * 10^(-1), 2.5130 * 10^(-1)],
        [5, 0, pr6_6, 8.5837, 8.1790, 7.2178, 6.1260, 5.8066, 5.1466, 4.3021, 3.5833, 2.9767, 2.8543, 2.4680, 2.0435, 1.6903, 1.3972, 1.1545, 9.5356 * 10^(-1), 7.8745 * 10^(-1), 6.5017 * 10^(-1), 5.9332 * 10^(-1), 5.3677 * 10^(-1)],
        [5 + 1 // 3, 0, pr7_7, 9.9136, 9.5404, 8.6113, 7.4967, 7.1643, 6.4685, 5.5579, 4.7607, 4.0680, 3.9257, 3.4697, 2.9551, 2.5142, 2.1374, 1.8161, 1.5423, 1.3095, 1.1115, 1.0278, 9.4336 * 10^(-1)],
        [5 + 2 // 3, 0, pr8_8, 1.1286 * 10^(1), 1.0908 * 10^(1), 9.9972, 8.8678, 8.5260, 7.8037, 6.8430, 5.9846, 5.2227, 5.0640, 4.5498, 3.9582, 3.4398, 2.9867, 2.5917, 2.2478, 1.9488, 1.6890, 1.5774, 1.4636],
        [6 + 1 // 3, 0, pr9_9, 1.2635 * 10^(1), 1.2262 * 10^(1), 1.1376 * 10^(1), 1.0238 * 10^(1), 9.8887, 9.1462, 8.1465, 7.2396, 6.4213, 6.2492, 5.6866, 5.0293, 4.4433, 3.9222, 3.4599, 3.0504, 2.6882, 2.3682, 2.2289, 2.0858],
        # [7, 0, pr10_10, 1.4002 * 10^(1), 1.3621 * 10^(1), 1.2750 * 10^(1), 1.1605 * 10^(1), 1.1251 * 10^(1), 1.0493 * 10^(1), 9.4621, 8.5158, 7.6513, 7.4679, 6.8648, 6.1516, 5.5069, 4.9257, 4.4027, 3.9331, 3.5119, 3.1347, 2.9688, 2.7971],
        # [7, 0, pr11_11, 1.5360 * 10^(1), 1.4972 * 10^(1), 1.4119 * 10^(1), 1.2971 * 10^(1), 1.2613 * 10^(1), 1.1842 * 10^(1), 1.0786 * 10^(1), 9.8071, 8.9041, 8.7113, 8.0739, 7.3129, 6.6173, 5.9830, 5.4058, 4.8816, 4.4060, 3.9753, 3.7843, 3.5855],
        # [7, 0, pr12_12, 1.6724 * 10^(1), 1.6327 * 10^(1), 1.5484 * 10^(1), 1.4335 * 10^(1), 1.3973 * 10^(1), 1.3191 * 10^(1), 1.2115 * 10^(1), 1.1109 * 10^(1), 1.0174 * 10^(1), 9.9731, 9.3064, 8.5042, 7.7643, 7.0833, 6.4578, 5.8842, 5.3589, 4.8786, 4.6642, 4.4399],
        [7 + 1 // 3, 0, pr13_13, 1.8103 * 10^(1), 1.7684 * 10^(1), 1.6846 * 10^(1), 1.5696 * 10^(1), 1.5331 * 10^(1), 1.4542 * 10^(1), 1.3448 * 10^(1), 1.2419 * 10^(1), 1.1456 * 10^(1), 1.1249 * 10^(1), 1.0557 * 10^(1), 9.7191, 8.9404, 8.2182, 7.5495, 6.9314, 6.3610, 5.8351, 5.5988, 5.3508],
        # [8 + 1 // 3, 0, pr14_14, 1.9451 * 10^(1), 1.9032 * 10^(1), 1.8205 * 10^(1), 1.7055 * 10^(1), 1.6689 * 10^(1), 1.5892 * 10^(1), 1.4783 * 10^(1), 1.3735 * 10^(1), 1.2749 * 10^(1), 1.2535 * 10^(1), 1.1822 * 10^(1), 1.0953 * 10^(1), 1.0140 * 10^(1), 9.3812, 8.6739, 8.0158, 7.4042, 6.8365, 6.5801, 6.3101],
        # [8 + 1 // 3, 0, pr15_15, 2.0818 * 10^(1), 2.0386 * 10^(1), 1.9562 * 10^(1), 1.8413 * 10^(1), 1.8045 * 10^(1), 1.7242 * 10^(1), 1.6120 * 10^(1), 1.5056 * 10^(1), 1.4048 * 10^(1), 1.3830 * 10^(1), 1.3097 * 10^(1), 1.2201 * 10^(1), 1.1358 * 10^(1), 1.0567 * 10^(1), 9.8254, 9.1311, 8.4821, 7.8761, 7.6013, 7.3109],
        # [8 + 1 // 3, 0, pr16_16, 2.2227 * 10^(1), 2.1756 * 10^(1), 2.0917 * 10^(1), 1.9769 * 10^(1), 1.9399 * 10^(1), 1.8591 * 10^(1), 1.7459 * 10^(1), 1.6380 * 10^(1), 1.5354 * 10^(1), 1.5131 * 10^(1), 1.4382 * 10^(1), 1.3462 * 10^(1), 1.2592 * 10^(1), 1.1772 * 10^(1), 1.0999 * 10^(1), 1.0272 * 10^(1), 9.5893, 8.9483, 8.6564, 8.3473],
        # [9, 0, pr17_17, 2.3596 * 10^(1), 2.3109 * 10^(1), 2.2270 * 10^(1), 2.1124 * 10^(1), 2.0752 * 10^(1), 1.9940 * 10^(1), 1.8798 * 10^(1), 1.7706 * 10^(1), 1.6665 * 10^(1), 1.6438 * 10^(1), 1.5674 * 10^(1), 1.4732 * 10^(1), 1.3839 * 10^(1), 1.2993 * 10^(1), 1.2192 * 10^(1), 1.1435 * 10^(1), 1.0722 * 10^(1), 1.0048 * 10^(1), 9.7409, 9.4144],
        # [9, 0, pr18_18, 2.4935 * 10^(1), 2.4451 * 10^(1), 2.3621 * 10^(1), 2.2477 * 10^(1), 2.2105 * 10^(1), 2.1288 * 10^(1), 2.0138 * 10^(1), 1.9035 * 10^(1), 1.7979 * 10^(1), 1.7749 * 10^(1), 1.6972 * 10^(1), 1.6011 * 10^(1), 1.5096 * 10^(1), 1.4227 * 10^(1), 1.3401 * 10^(1), 1.2617 * 10^(1), 1.1875 * 10^(1), 1.1172 * 10^(1), 1.0851 * 10^(1), 1.0508 * 10^(1)]
    ]
)

matlab_lut = reduce(hcat,
    [
        [Inf, Inf, Inf, 0.0, -1.0, -2.0, -3.0, -3.31133, -4.0, -5.0, -6.0, -7.0, -7.22472, -8.0, -9.0, -1 * 10^(1), -1.1 * 10^(1), -1.2 * 10^(1), -1.3 * 10^(1), -1.4 * 10^(1), -1.5 * 10^(1), -1.54775 * 10^(1), -1.6 * 10^(1)],
        [3 + 1 // 3, 0, pr3_3, 4.4808, 3.9709, 2.9793, 2.0971, 1.8718, 1.4500, 9.9496 * 10^(-1), 6.8016 * 10^(-1), 4.6413 * 10^(-1), 4.2587 * 10^(-1), 3.1644 * 10^(-1), 2.1566 * 10^(-1), 1.4695 * 10^(-1), 1.0013 * 10^(-1), 6.8218 * 10^(-2), 4.6477 * 10^(-2), 3.1665 * 10^(-2), 2.1573 * 10^(-2), 1.7961 * 10^(-2), 1.4697 * 10^(-2)],
        [4 + 1 // 3, 0, pr5_5, 7.1999, 6.7906, 5.8144, 4.7596, 4.4590, 3.8495, 3.0946, 2.4777, 1.9783, 1.8802, 1.5766, 1.2550, 9.9825 * 10^(-1), 7.9362 * 10^(-1), 6.3074 * 10^(-1), 5.0119 * 10^(-1), 3.9819 * 10^(-1), 3.1634 * 10^(-1), 2.8342 * 10^(-1), 2.5130 * 10^(-1)],
        [5 + 1 // 3, 0, pr7_7, 9.9136, 9.5404, 8.6113, 7.4967, 7.1643, 6.4685, 5.5579, 4.7607, 4.0680, 3.9257, 3.4697, 2.9551, 2.5142, 2.1374, 1.8161, 1.5423, 1.3095, 1.1115, 1.0278, 9.4336 * 10^(-1)],
        [6 + 1 // 3, 0, pr9_9, 1.2635 * 10^(1), 1.2262 * 10^(1), 1.1376 * 10^(1), 1.0238 * 10^(1), 9.8887, 9.1462, 8.1465, 7.2396, 6.4213, 6.2492, 5.6866, 5.0293, 4.4433, 3.9222, 3.4599, 3.0504, 2.6882, 2.3682, 2.2289, 2.0858],
        [7 + 1 // 3, 0, pr13_13, 1.8103 * 10^(1), 1.7684 * 10^(1), 1.6846 * 10^(1), 1.5696 * 10^(1), 1.5331 * 10^(1), 1.4542 * 10^(1), 1.3448 * 10^(1), 1.2419 * 10^(1), 1.1456 * 10^(1), 1.1249 * 10^(1), 1.0557 * 10^(1), 9.7191, 8.9404, 8.2182, 7.5495, 6.9314, 6.3610, 5.8351, 5.5988, 5.3508]
    ]
)