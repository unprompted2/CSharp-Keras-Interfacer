﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Wineforever.String;
namespace Wineforever.Neuralnet
{
    #region 被弃用
    /*
    public class Neuron //定义神经元
    {
        //公共字段
        public Neuron(dynamic input = null, string name = null, string activation = "relu", string optimizer = "GD")//构造神经元
        {
            //---- 生成检索码 ----
            var GUID = Guid.NewGuid();//全局唯一标识符
            string ID = GUID.ToString().Replace("-", "").ToUpper();//转换成大写
            //---- ----
            //---- 初始化权重 ----
            if (input != null)
            {
                Type input_type = input.GetType();
                if (input_type == typeof(float))
                    Initialization(0);
                else if (input_type == typeof(List<Neuron>))
                    Initialization(input.Count);
                else if (input_type == typeof(List<float>))
                    Initialization(input.Count);
                else if (input_type == typeof(FlattenLayer))
                    Initialization(0);
            }
            else
                Initialization(0);
            //---- ----
            //---- 创建神经元 ----
            this.Name = name == null ? "Undefined" : name;
            this.ID = ID;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            //---- ----
        }
        //私有字段
        internal string Name { get; set; }//标识名
        internal string ID { get; }//检索码
        internal dynamic Input { get; set; }//多输入
        internal string Activation { get; }//激励器
        internal string Optimizer { get; }//优化器
        internal DenseLayer Father { get; set; }//父对象
        internal List<float> Weights { get; set; }//权重
        internal float Baisc { get; set; }//偏置量
        private bool __isHidden__ = false;
        //公共方法
        public float GetOutput()//返回当前神经元的输出
        {
            float output = 0;
            //---- 判断输入类型 ----
            Type input_type = this.Input.GetType();
            if (input_type == typeof(float))
            {
                return this.Input;
            }
            if (input_type == typeof(List<Neuron>))
            {
                if (this.Optimizer.Contains("dropout") && this.Father.Father.Rand(0, 1) <= this.Father.Father.dropout_rate)
                {
                    output = 0;
                    this.__isHidden__ = true;
                }
                else
                {
                    List<Neuron> input_neurons = this.Input;//得到输入神经元
                    for (int i = 0; i < input_neurons.Count; i++)
                    {
                        Neuron neuron = input_neurons[i];
                        var neuron_output = neuron.GetOutput();
                        output += neuron_output * this.Weights[i];
                    }
                    output += this.Baisc;
                }
            }
            else if (input_type == typeof(List<float>))
            {
                if (this.Optimizer.Contains("dropout") && this.Father.Father.Rand(0, 1) <= this.Father.Father.dropout_rate)
                {
                    output = 0;
                    this.__isHidden__ = true;
                }
                else
                {
                    List<float> input = this.Input;
                    for (int i = 0; i < input.Count; i++)
                    {
                        output += input[i] * this.Weights[i];
                    }
                    output += this.Baisc;
                }
            }
            else if (input_type == typeof(FlattenLayer))
            {
                if (this.Optimizer.Contains("dropout") && this.Father.Father.Rand(0, 1) <= this.Father.Father.dropout_rate)
                {
                    output = 0;
                    this.__isHidden__ = true;
                }
                else
                {
                    List<float> input = this.Input.GetOutput();
                    for (int i = 0; i < input.Count; i++)
                    {
                        output += input[i] * this.Weights[i];
                    }
                    output += this.Baisc;
                }
            }
            return activation(output, this.Activation);
        }
        public void Connect(Neuron neuron)//将当前神经元的输出端连接到指定神经元的输入端
        {
            List<Neuron> inputs = neuron.Input;
            inputs.Add(this);
            neuron.Input = inputs;
        }
        public void Disconnect(Neuron neuron)//切断与指定神经元的连接
        {
            List<Neuron> inputs = neuron.Input;
            inputs.Remove(this);
            neuron.Input = inputs;
        }
        public void Update(float error, string optimizer)//更新权重
        {
            var Weights = this.Weights;
            var Basic = this.Baisc;
            float learning_rate = this.Father.Father.learning_rate;
            List<float> deltas = new List<float>();
            if (!this.__isHidden__)
            {
                for (int i = 0; i < Weights.Count; i++)
                {
                    float input = 0;
                    string activation = "normal";
                    if (this.Input.GetType() == typeof(List<Neuron>))
                    {
                        input = this.Input[i].GetOutput();
                        activation = this.Input[i].Activation;
                    }
                    else if (this.Input.GetType() == typeof(List<float>))
                    {
                        input = this.Input[i];
                        activation = "normal";
                    }
                    float delta = 0;
                    //优化器
                    if (optimizer.Contains("GD"))//标准梯度下降法
                    {
                        delta = error * input * autodiff(this.GetOutput(), activation) * learning_rate;
                    }
                    else if (optimizer.Contains("BGD"))//批量梯度下降法
                    {
                        delta = this.Father.Father.__loss__ * input * autodiff(this.GetOutput(), activation) * learning_rate;
                    }
                    else if (optimizer.Contains("SGD"))//随机梯度下降法
                    {
                        float SGD = this.Father.Father.errors[(int)this.Father.Father.Rand(0, (this.Father.Father.Layers.Last() as DenseLayer).Neurons.Count)] * input * autodiff(this.GetOutput(), activation);
                        delta = learning_rate * SGD;
                        Console.WriteLine("SGD");
                    }
                    deltas.Add(delta);
                    //---- 更新权重 ----
                    Weights[i] -= deltas[i];
                    //---- ----
                }
                //---- 更新偏置量 ----
                Basic -= error * learning_rate;
                //---- ----
                this.Weights = Weights;
                this.Baisc = Baisc;
                //---- 找到前级神经元，递归更新权重 ----
                if (this.Input.GetType() == typeof(List<Neuron>))
                {
                    var pre_layer_neurons = this.Input;
                    for (int i = 0; i < pre_layer_neurons.Count; i++)
                        if (pre_layer_neurons[i].Input.GetType() == typeof(List<Neuron>) || pre_layer_neurons[i].Input.GetType() == typeof(List<float>))
                            pre_layer_neurons[i].Update(deltas[i], pre_layer_neurons[i].Optimizer);
                }
            }
            else { this.__isHidden__ = false; }//取消隐藏
        }
        public void Save()//输出权重信息
        {
            string FileName = AppDomain.CurrentDomain.BaseDirectory + "neurons\\" + this.Name + ".txt";
            Dictionary<string, string> Data = new Dictionary<string, string>();
            Data["Name"] = this.Name;
            Data["ID"] = this.ID;
            Data["Activation"] = this.Activation;
            Data["Optimizer"] = this.Optimizer;
            var neurons = this.Input;
            for (int i = 0; i < neurons.Count; i++)
            {
                Data["Weight" + i] = this.Weights[i].ToString();
            }
            Data["Basic"] = this.Baisc.ToString();
            Wineforever.String.client.SaveToList(Data, FileName);
        }
        public void SetWeight(float Value)
        {
            for (int i = 0; i < this.Weights.Count; i++)
            {
                this.Weights[i] = Value;
            }
        }
        public void SetWeight(List<float> Values)
        {
            for (int i = 0; i < this.Weights.Count; i++)
            {
                this.Weights[i] = Values[i];
            }
        }
        public void SetWeight(List<string> Values)
        {
            for (int i = 0; i < this.Weights.Count; i++)
            {
                this.Weights[i] = float.Parse(Values[i]);
            }
        }
        public void SetBias(float Value)
        {
            this.Baisc = Value;
        }
        public void SetBias(string Value)
        {
            this.Baisc = float.Parse(Value);
        }
        public void Print()
        {
            for (int i = 0; i < this.Weights.Count; i++)
                Console.WriteLine("weights:{0}", this.Weights[i]);
            for (int i = 0; i < this.Input.Count; i++)
                Console.WriteLine("input:{0}", this.Input[i]);
            Console.WriteLine("bias:{0}", this.Baisc);
            Console.WriteLine("output:{0}", this.GetOutput());
        }
        //私有方法
        //参数初始化
        internal void Initialization(int input_Count)
        {
            var Weights = new List<float>();
            for (int i = 0; i < input_Count; i++)
            {
                Weights.Add(1);
            }
            this.Weights = Weights;
            this.Baisc = 0;
        }
        //调用外部激励器
        private float activation(float x, string func)
        {
            if (func == "relu") return x > 0 ? x : 0;
            else if (func == "normal") return x;
            else if (func == "sigmoid") return 1.0f / (1.0f + (float)Math.Pow(Math.E, -x));
            else if (func == "leaky relu") return x > 0 ? x : 0.01f * x;
            else if (func == "softmax") return x;
            else return 0;
        }
        //自动求导
        private float autodiff(float x, string func)
        {
            if (func == "relu") return x > 0 ? 1 : 0;
            else if (func == "normal") return 1;
            else if (func == "sigmoid") return activation(x, "sigmoid") * (1 - activation(x, "sigmoid"));
            else if (func == "leaky relu") return x > 0 ? 1 : 0.01f;
            else if (func == "softmax") return 1;
            else
            {
                float d = 1e-4f;//数值微分
                return (activation(x + d, func) - activation(x, func)) / d;
            }
        }
        //---- ----
    }
    */
    #endregion
    #region 被弃用
    /*
    public class Kernel//定义卷积核
    {
        //公共字段
        public Kernel(int size, dynamic input = null, string name = null, string padding = "same", int step = 1, string activation = "relu", string optimizer = "GD")//构造卷积核
        {
            //---- 生成检索码 ----
            var GUID = Guid.NewGuid();//全局唯一标识符
            string ID = GUID.ToString().Replace("-", "").ToUpper();//转换成大写
            //---- ----
            //---- 初始化权重 ----
            if (input != null)
            {
                Type input_type = input.GetType();
                if (input_type == typeof(float[,]))
                {
                    Initialization(size, 1);
                }
                else if (input_type == typeof(List<float[,]>))
                {
                    Initialization(size, input.Count());
                }
                else if (input_type == typeof(ConvLayer))
                {
                    Initialization(size, (input as ConvLayer).Kernels.Count());
                }
            }
            else
            {
                Initialization(size, 4);
            }
            //---- ----
            //---- 创建卷积核 ----
            this.Name = name == null ? "Undefined" : name;
            this.ID = ID;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            this.Size = size;
            this.Padding = padding;
            this.Step = step;
            //---- ----
        }
        //私有字段
        internal string Name { get; set; }//标识名
        internal string ID { get; }//检索码
        internal dynamic Input { get; set; }//多输入
        internal string Activation { get; }//激励器
        internal string Optimizer { get; }//优化器
        internal ConvLayer Father { get; set; }//父对象
        internal float[,,] Weights { get; set; }//权重
        internal float Bias { get; set; }//偏置量
        internal int Size { get; set; }//尺寸
        internal string Padding { get; }//扩充模式
        internal int Step { get; set; }//步长
        //公有方法
        public dynamic GetOutput()
        {
            dynamic output = null;
            // --- 判断输入类型 --- 
            Type input_type = this.Input.GetType();
            if (input_type == typeof(float[,]))
            {
                //单通道
                output = Wineforever.Mathematics.client.conv(Input, Weights, Bias, Step, Activation, Padding);
                return output;
            }
            else if (input_type == typeof(List<float[,]>))
            {
                //多通道
                output = Wineforever.Mathematics.client.conv(Input, Weights, Bias, Step, Activation, Padding);
                return output;
            }
            else if (input_type == typeof(ConvLayer))
            {
                var input = this.Input.GetOutput();
                if (input.GetType() == typeof(float[,]))
                {
                    output = Wineforever.Mathematics.client.conv(input, Weights, Bias, Step, Activation, Padding);
                }
                else if (input.GetType() == typeof(List<float[,]>))
                {
                    output = Wineforever.Mathematics.client.conv(input, Weights, Bias, Step, Activation, Padding);
                }
                return output;
            }
            return -1;
        }
        public void SetWeight(float Value)
        {
            int width = Weights.GetLength(0);
            int height = Weights.GetLength(1);
            int channel = Weights.GetLength(2);
            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    for (int k = 0; k < channel; k++)
                        Weights[i, j, k] = Value;
        }
        public void SetWeight(float[,,] Values)
        {
            int width = Weights.GetLength(0);
            int height = Weights.GetLength(1);
            int channel = Weights.GetLength(2);
            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    for (int k = 0; k < channel; k++)
                        Weights[i, j, k] = Values[i, j, k];
        }
        public void SetWeight(float Value, int width_index, int height_index, int channel_index)
        {
            Weights[width_index, height_index, channel_index] = Value;
        }
        public void SetBias(float Value)
        {
            Bias = Value;
        }
        //私有方法
        internal void Initialization(int size, int channel)
        {
            var Weights = new float[size, size, channel];
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    for (int k = 0; k < channel; k++)
                        Weights[i, j, k] = 1.0f;
            this.Weights = Weights;
            this.Bias = 0;
        }
        //调用外部激励器
        private float activation(float x, string func)
        {
            if (func == "relu") return x > 0 ? x : 0;
            else if (func == "normal") return x;
            else if (func == "sigmoid") return 1.0f / (1.0f + (float)Math.Pow(Math.E, -x));
            else if (func == "leaky relu") return x > 0 ? x : 0.01f * x;
            else if (func == "softmax") return x;
            else return 0;
        }
    }
    */
    #endregion
    //---- 层类 ----
    public abstract class Layer
    {
        //私有字段
        internal string Name { get; set; }//标识名
        internal dynamic Input { get; set; }//输入
        internal string Activation { get; set; }
        internal string Optimizer { get; set; }
        internal System Father { get; set; }
        //公有方法
        public dynamic GetOutput()
        {
            dynamic Output = null;
            if (this.GetType() == typeof(DenseLayer))
            {
                Output = (this as DenseLayer).GetOutput();
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                Output = (this as ConvLayer).GetOutput();
            }
            else if (this.GetType() == typeof(FlattenLayer))
            {
                Output = (this as FlattenLayer).GetOutput();
            }

            return Output;
        }
        public void LoadKernel(string FileName)
        {
            if (this.GetType() == typeof(DenseLayer))
            {
                (this as DenseLayer).LoadWeights(FileName);
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                (this as ConvLayer).LoadKernel(FileName);
            }
        }
        public void LoadBias(string FileName)
        {
            if (this.GetType() == typeof(DenseLayer))
            {
                (this as DenseLayer).LoadBias(FileName);
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                (this as ConvLayer).LoadBias(FileName);
            }
        }
        public void Print()
        {
            Console.WriteLine("=======================");
            Console.WriteLine("========= {0} =========", Name);
            Console.WriteLine("=======================");
            Console.WriteLine("Input: {0}", Input);
            Console.WriteLine("Activation: {0}", Activation);
            Console.WriteLine("Optimizer: {0}", Optimizer);
            Console.WriteLine("Father: {0}", Father);
            if (this.GetType() == typeof(DenseLayer))
            {
                Console.WriteLine("---- {0} ----", (this as DenseLayer).Name + "{Weights}");
                Wineforever.Mathematics.client.print((this as DenseLayer).__weights__);
                Console.WriteLine("---- Bias ----");
                Wineforever.Mathematics.client.print((this as DenseLayer).__bias__);
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                Console.WriteLine("Kernel: ");
                for (int i = 0; i < (this as ConvLayer).__kernelNum__; i++)
                {
                    Console.WriteLine("---- {0} ----", (this as ConvLayer).Name + "{Weights}");
                    Wineforever.Mathematics.client.print((this as ConvLayer).__weights__[i]);
                    Console.WriteLine("Bias: {0}", (this as ConvLayer).__bias__[i]);
                }
            }
        }
    }
    //全连接层
    public class DenseLayer : Layer
    {
        //私有字段
        internal string __kernelPath__ = "";
        internal string __biasPath__ = "";
        internal float[,] __weights__ = null;
        internal float[] __bias__ = null;
        internal int __neurons__ = 0;
        internal int __inputNum__ = 0;
        internal float[] __input__ = null;
        internal dynamic __neuron_Weights_data__ = null;
        internal dynamic __neuron_Bias_data__ = null;
        //公有字段
        public DenseLayer(int neuron_num, dynamic input = null, string name = null, string activation = "relu", string optimizer = "GD")
        {
            //---- 创建层 ----
            this.Name = name == null ? "Undefined" : name;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            //---- ----
            //---- 创建神经元 ----
            this.__weights__ = null;
            this.__neurons__ = neuron_num;
            //---- ----
            //判断输入类型
            if (input != null)
            {
                if (input.GetType() == typeof(DenseLayer))//非头层
                {
                    DenseLayer Input = input;
                    this.__inputNum__ = Input.__neurons__;
                    this.__weights__ = new float[__inputNum__, __neurons__];
                    this.__bias__ = new float[__neurons__];
                }
                if (input.GetType() == typeof(FlattenLayer))
                {
                    FlattenLayer Input = input;
                    this.__inputNum__ = Input.__outputNum__;
                    this.__weights__ = new float[__inputNum__, __neurons__];
                    this.__bias__ = new float[__neurons__];
                }
                else if (input.GetType() == typeof(List<float>))//头层
                {
                    List<float> Input = input;
                    this.__inputNum__ = Input.Count;
                    this.__weights__ = new float[__inputNum__, __neurons__];
                    this.__bias__ = new float[__neurons__];
                }
            }
            else
            {
                for (int i = 0; i < neuron_num; i++)
                {
                    DenseLayer Input = input;
                    this.__inputNum__ = Input.__neurons__;
                    this.__weights__ = null;
                    this.__bias__ = new float[__neurons__];
                }
            }
        }
        //公有方法
        public new List<float> GetOutput()
        {
            //耗时计算
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            List<float> Output = new List<float>();
            //---- 判断输入类型 ----
            Type input_type = this.Input.GetType();
            if (input_type == typeof(List<float>))
            {
                List<float> input = this.Input;
                var _input = input.ToArray();
                var res = Wineforever.Mathematics.client.dense(_input, __weights__, __bias__, Activation);
                Output = res.ToList();
            }
            else if (input_type == typeof(FlattenLayer))
            {
                FlattenLayer input = this.Input;
                var _input = input.GetOutput().ToArray();
                var res = Wineforever.Mathematics.client.dense(_input, __weights__, __bias__, Activation);
                Output = res.ToList();
            }
            else if (input_type == typeof(DenseLayer))
            {
                DenseLayer input = this.Input;
                var _input = input.GetOutput().ToArray();
                var res = Wineforever.Mathematics.client.dense(_input, __weights__, __bias__, Activation);
                Output = res.ToList();
            }
            if (this.Activation == "softmax")
            {
                float sum = Output.Sum(i => (float)Math.Pow(Math.E, i - Output.Max()));
                Output = Output.Select(i => (float)Math.Pow(Math.E, i - Output.Max()) / sum).ToList();
            }
            //打印
            stopwatch.Stop();
            if(this.Father.__debug__) Console.WriteLine("{0}用时(毫秒)：{1}", this.Name, stopwatch.ElapsedMilliseconds);
            return Output;
        }
        public void LoadWeights(string FileName)
        {
            __neuron_Weights_data__ = __neuron_Weights_data__ == null ? Wineforever.String.client.Deserialization(FileName) : __neuron_Weights_data__;
            this.__weights__ = __neuron_Weights_data__;
            this.Father.__isInitialization__ = true;
            this.__kernelPath__ = FileName;
        }
        public new void LoadBias(string FileName)
        {
            __neuron_Bias_data__ = __neuron_Bias_data__ == null ? Wineforever.String.client.Deserialization(FileName) : __neuron_Bias_data__;
            this.__bias__ = __neuron_Bias_data__;
            this.Father.__isInitialization__ = true;
            this.__biasPath__ = FileName;
        }
    }
    //卷积层
    public class ConvLayer : Layer
    {
        //私有字段
        internal string Paddding { get; }
        internal int Step { get; }
        internal int Size { get; }
        internal string __kernelPath__ = "";
        internal string __biasPath__ = "";
        internal List<float[,,]> __weights__ = null;
        internal List<float> __bias__ = null;
        internal int __kernelNum__ = 0;
        internal dynamic __kernel_Weights_data__ = null;
        internal dynamic __kernel_Bias_data__ = null;
        //公有字段
        public ConvLayer(int kernel_num, int kernel_size, int step = 1, dynamic input = null, string name = null, string padding = "same", string activation = "relu", string optimizer = "GD")
        {
            //---- 创建层 ----
            this.Name = name == null ? "Undefined" : name;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            this.Paddding = padding;
            this.Step = step;
            this.Size = kernel_size;
            this.__kernelNum__ = kernel_num;
            //---- ----
            //---- 创建卷积核 ----
            this.__weights__ = new List<float[,,]>();
            this.__bias__ = new List<float>();
            //---- ----
        }
        //公有方法
        public new List<float[,]> GetOutput()
        {
            //耗时计算
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            List<float[,]> Output = new List<float[,]>();
            if (this.Input.GetType() == typeof(List<float[,]>))//首层
            {
                var input = Input as List<float[,]>;
                for (int i = 0; i < __weights__.Count; i++)
                {
                    var output = Wineforever.Mathematics.client.conv(input, __weights__[i], __bias__[i], Step, Activation, Paddding);
                    Output.Add(output);
                };
            }
            else if (Input.GetType() == typeof(ConvLayer))
            {
                var input = (Input as ConvLayer).GetOutput();
                for (int i = 0; i < __weights__.Count; i++)
                {
                    var output = Wineforever.Mathematics.client.conv(input, __weights__[i], __bias__[i], Step, Activation, Paddding);
                    Output.Add(output);
                };
            }
            //打印
            stopwatch.Stop();
            if (this.Father.__debug__) Console.WriteLine("{0}用时(毫秒)：{1}", this.Name, stopwatch.ElapsedMilliseconds);
            return Output;
        }
        public void LoadKernel(string FileName, string mod = "channel_last")
        {
            this.__weights__ = new List<float[,,]>();
            if (this.Input != null)
            {
                if (mod == "channel_first")
                {
                    __kernel_Weights_data__ = __kernel_Weights_data__ == null ? Wineforever.String.client.Deserialization(FileName) : __kernel_Weights_data__;
                    if (this.Input.G