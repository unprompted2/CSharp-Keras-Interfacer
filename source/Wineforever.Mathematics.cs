
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wineforever.Mathematics
{
    public class client
    {
        #region Dense
        public static float[] dense(float[] input, float[,] weights, float[] bias, string activator = "relu")
        {
            int input_len = input.Length;
            int output_len = weights.GetLength(1);
            var output = new float[output_len];
            for (int i = 0; i < output_len; i++)
            {
                for (int j = 0; j < input_len; j++)
                    output[i] += input[j] * weights[j, i];
                output[i] += bias[i];
                output[i] = activation(output[i], activator);
            }
            return output;
        }
        #endregion
        #region Convolution
        public static float[,] conv(List<float[,]> input, float[,,] kernel, float bias = 0, int step = 1, string activator = "relu", string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int kernel_channel = kernel.GetLength(2);
            int width = input[0].GetLength(0);
            int height = input[0].GetLength(1);
            int input_channel = input.Count();
            List<float[,]> input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((float)width / (float)step);
                output_height = (int)System.Math.Ceiling((float)height / (float)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            float[,] output = new float[output_width, output_height];
            var res = new float[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    float channel_sum = 0;
                    for (int u = 0; u < input_channel; u++)
                    {
                        int X = i * step;
                        int Y = j * step;
                        for (int m = 0; m < kernel_width; m++)
                            for (int n = 0; n < kernel_height; n++)
                                channel_sum += input_padding[u][X + m, Y + n] * kernel[m, n, u];

                    }
                    channel_sum += bias;
                    output[i, j] = activation(channel_sum, activator);
                }
            return output;
        }
        #endregion
        #region Padding
        public static List<float[,]> padding(List<float[,]> input, float[,,] kernel, int step = 1, string padding_mod = "same")
        {
            List<float[,]> output = new List<float[,]>(); ;
            int width = input[0].GetLength(0);
            int height = input[0].GetLength(1);
            int input_channel = input.Count;
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int kernel_channel = kernel.GetLength(2);
            if (padding_mod == "same")
            {
                int _width = (int)System.Math.Ceiling((float)width / (float)step);
                int _height = (int)System.Math.Ceiling((float)height / (float)step);
                int pad_needed_width = (_width - 1) * step + kernel_width - width;
                int pad_left = pad_needed_width / 2;
                int pad_right = pad_needed_width - pad_left;
                int pad_needed_height = (_height - 1) * step + kernel_height - height;
                int pad_top = pad_needed_height / 2;
                int pad_bottom = pad_needed_height - pad_top;
                for (int n = 0; n < input_channel; n++)
                {
                    var res = new float[pad_left + width + pad_right, pad_top + height + pad_bottom];
                    for (int i = pad_left; i < width + pad_left; i++)
                        for (int j = pad_bottom; j < height + pad_bottom; j++)
                            res[i, j] = input[n][i - pad_left, j - pad_bottom];
                    output.Add(res);
                }
            }
            return output;
        }
        #endregion
        public static double[,] sum(List<double[,]> inputs)
        {
            var width = inputs.Count > 0 ? inputs[0].GetLength(0) : 0;
            var height = inputs.Count > 0 ? inputs[0].GetLength(1) : 0;
            double[,] output = new double[width, height];
            for (int n = 0; n < inputs.Count; n++)
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++)
                        output[i, j] += inputs[n][i, j];
            return output;
        }
        #region Activation
        public static float activation(float x, string func)
        {
            if (func == "relu") return x > 0 ? x : 0;
            else if (func == "normal") return x;
            else if (func == "sigmoid") return 1.0f / (1.0f + (float)Math.Pow(Math.E, -x));
            else if (func == "leaky relu") return x > 0 ? x : 0.01f * x;
            else if (func == "softmax") return x > 0 ? x : 0;
            else return 0;
        }
        #endregion
        #region Print
        public static void print(double[] input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.Length; i++)
            {
                sb.Append(input[i] + "  ");
                if ((i + 1) % 10 == 0) sb.Append("\r\n");
            }
            sb.Append("\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(double[,] input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append(input[i, j] + " ");
                }
                sb.Append("\r\n");
            }
            Console.Write(sb.ToString());
        }
        public static void print(double[,,] input)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < input.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append("[");
                    for (int k=0;k<input.GetLength(2);k++)
                    sb.Append(input[i, j,k] + " ");
                    sb.Append("]");
                }
                sb.Append("]");
            }
            sb.Append("]\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(double[,,,] input)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < input.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append("[");
                    for (int k = 0; k < input.GetLength(2); k++)
                    {
                        sb.Append("[");
                        for (int l = 0; l < input.GetLength(3); l++)
                            sb.Append(input[i, j, k, l] + " ");
                        sb.Append("]");
                    }
                    sb.Append("]");
                }
                sb.Append("]");
            }
            sb.Append("]\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(List<double[,]> input)
        {
            StringBuilder sb = new StringBuilder();
            for (int n = 0; n < input.Count; n++)
            {
                sb.Append("\r\n---- Depth:" + n + " ----\r\n");
                for (int i = 0; i < input[n].GetLength(0); i++)
                {
                    for (int j = 0; j < input[n].GetLength(1); j++)
                    {
                        sb.Append(input[n][i, j] + " ");
                    }
                    sb.Append("\r\n");
                }
                sb.Append("---- ------- ----\r\n");
            }
            Console.Write(sb.ToString());
        }
        public static void print(List<double> input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.Count; i++)
            {
                sb.Append(input[i] + "  ");
                if ((i + 1) % 10 == 0) sb.Append("\r\n");
            }
            sb.Append("\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(float[] input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.Length; i++)
            {
                sb.Append(input[i] + "  ");
                if ((i + 1) % 10 == 0) sb.Append("\r\n");
            }
            sb.Append("\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(float[,] input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append(input[i, j] + " ");
                }
                sb.Append("\r\n");
            }
            Console.Write(sb.ToString());
        }
        public static void print(float[,,] input)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < input.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append("[");
                    for (int k = 0; k < input.GetLength(2); k++)
                        sb.Append(input[i, j, k] + " ");
                    sb.Append("]");
                }
                sb.Append("]");
            }
            sb.Append("]\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(float[,,,] input)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < input.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append("[");
                    for (int k = 0; k < input.GetLength(2); k++)
                    {
                        sb.Append("[");
                        for (int l = 0; l < input.GetLength(3); l++)
                            sb.Append(input[i, j, k, l] + " ");
                        sb.Append("]");
                    }
                    sb.Append("]");
                }
                sb.Append("]");
            }
            sb.Append("]\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(List<float[,]> input)
        {
            StringBuilder sb = new StringBuilder();
            for (int n = 0; n < input.Count; n++)
            {
                sb.Append("\r\n---- Depth:" + n + " ----\r\n");
                for (int i = 0; i < input[n].GetLength(0); i++)
                {
                    for (int j = 0; j < input[n].GetLength(1); j++)
                    {
                        sb.Append(input[n][i, j] + " ");
                    }
                    sb.Append("\r\n");
                }
                sb.Append("---- ------- ----\r\n");
            }
            Console.Write(sb.ToString());
        }
        public static void print(List<float> input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.Count; i++)
            {
                sb.Append(input[i] + "  ");
                if ((i + 1) % 10 == 0) sb.Append("\r\n");
            }
            sb.Append("\r\n");
            Console.Write(sb.ToString());
        }
        #endregion
        #region GetShape
        public static int[] get_shape(double[,] input)
        {
            int[] res = new int[2];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            Console.WriteLine("(" + res[0] + "," + res[1] + ")");
            return res;
        }
        public static int[] get_shape(double[,,] input)
        {
            int[] res = new int[3];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            res[2] = input.GetLength(2);
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + ")");
            return res;
        }
        public static int[] get_shape(double[,,,] input)
        {
            int[] res = new int[4];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            res[2] = input.GetLength(2);
            res[3] = input.GetLength(3);
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + "," + res[3] + ")");
            return res;
        }
        public static int[] get_shape(List<double[,]> input)
        {
            int[] res = new int[3];
            res[0] = input.Count;
            if (input.Count > 0)
                res[1] = input[0].GetLength(0);
            else res[1] = 0;
            if (input.Count > 0)
                res[2] = input[0].GetLength(1);
            else res[2] = 0;
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + ")");
            return res;
        }
        public static int[] get_shape(List<double> input)
        {
            int[] res = new int[1];
            res[0] = input.Count;
            Console.WriteLine("(" + res[0] + ")");
            return res;
        }
        public static int[] get_shape(float[,] input)
        {
            int[] res = new int[2];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            Console.WriteLine("(" + res[0] + "," + res[1] + ")");
            return res;
        }
        public static int[] get_shape(float[,,] input)
        {
            int[] res = new int[3];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            res[2] = input.GetLength(2);
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + ")");
            return res;
        }
        public static int[] get_shape(float[,,,] input)
        {
            int[] res = new int[4];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            res[2] = input.GetLength(2);
            res[3] = input.GetLength(3);
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + "," + res[3] + ")");
            return res;
        }
        public static int[] get_shape(List<float[,]> input)
        {
            int[] res = new int[3];
            res[0] = input.Count;
            if (input.Count > 0)
                res[1] = input[0].GetLength(0);
            else res[1] = 0;
            if (input.Count > 0)
                res[2] = input[0].GetLength(1);
            else res[2] = 0;
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + ")");
            return res;
        }
        public static int[] get_shape(List<float> input)
        {
            int[] res = new int[1];
            res[0] = input.Count;
            Console.WriteLine("(" + res[0] + ")");
            return res;
        }
        #endregion
    }
}