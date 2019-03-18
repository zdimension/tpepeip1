using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace tpepeip1
{
    class Program
    {
        static void Main(string[] args)
        {
            var app = new App(new Dictionary<string, string>());

            while (true)
            {
                app.AppLoop();
            }
        }
    }
}
