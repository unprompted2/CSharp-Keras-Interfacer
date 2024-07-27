using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wineforever.Mathematics
{
    public class client
    {
        public static double[,] para_conv(double[,] input, double[,] kernel, double bias = 0, int step = 1, string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,] input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            Parallel.For(0, output_width, i =>
            {
                Parallel.For(0, output_height, j =>
                {
                    int X = i * step;
                    int Y = j * step;
                    double sum = 0;
                    for (int m = 0; m < kernel_width; m++)
                        for (int n = 0; n < kernel_height; n++)
                            sum += input_padding[X + m, Y + n] * kernel[m,n];
                    output[i, j] = sum + bias;
                });
            });
            return output;
        }
        public static double[,] conv(double[,] input, double[,] kernel, double bias = 0, int step = 1, string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,] input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    int X = i * step;
                    int Y = j * step;
                    double sum = 0;
                    for (int m = 0; m < kernel_width; m++)
                        for (int n = 0; n < kernel_height; n++)
                            sum += input_padding[X + m, Y + n] * kernel[m,n];
                    output[i, j] = sum + bias;
                }
            return output;
        }
        public static double[,] conv(double[,] input, double[,] kernel, double bias = 0, int step = 1, string activator = "relu", string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,] input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    int X = i * step;
                    int Y = j * step;
                    double sum = 0;
                    for (int m = 0; m < kernel_width; m++)
                        for (int n = 0; n < kernel_height; n++)
                            sum += input_padding[X + m, Y + n] * kernel[m,n];
                    output[i, j] = activation(sum + bias, activator);
                }
            return output;
        }
        public static double[,] conv(List<double[,]> input, double[,,] kernel, double bias = 0, int step = 1, string activator = "relu", string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int kernel_channel = kernel.GetLength(2);
            int width = input[0].GetLength(0);
            int height = input[0].GetLength(1);
            int input_channel = input.Count();
            List<double[,]> input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            var res = new double[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    double channel_sum = 0;
                    //Input Channel == Kernel Channel
                    for (int u = 0; u < input_channel; u++)
                    {
                        int X = i * step;
                        int Y = j * step;
                        for (int m = 0; m < kernel_width; m++)
                            for (int n = 0; n < kernel_height; n++)
                                channel_sum += input_padding[u][X + m, Y + n] * kernel[m, n, u];
        