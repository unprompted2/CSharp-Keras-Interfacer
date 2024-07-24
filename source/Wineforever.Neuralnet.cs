using System;
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
                        delta = error * i