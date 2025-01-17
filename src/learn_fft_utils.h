/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once
#include <vector>

static void GenBitReverseOrder(size_t len, std::vector<size_t>& arr)
{
    for (size_t i = 0; i < len; i++)
    {
        arr[i] = 0;
        size_t idx = i;
        size_t step = len / 2;
        while (idx > 0)
        {
            if (idx % 2 == 1)
                arr[i] += step;
            idx /= 2;
            step /= 2;
        }
    }
}
