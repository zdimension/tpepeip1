using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace tpepeip1
{
    public static class Logger
    {
        public static void Wrap(string text, params string[] args)
        {
            Console.Write("[" + text.PadLeft(5, ' ') + "] " + string.Join(" ", args));
        }

        public static void Info(params string[] args)
        {
            Wrap("INFO", args);
        }

        public static void Error(params string[] args)
        {
            Wrap("ERROR", args);
        }

        public static void Warn(params string[] args)
        {
            Wrap("WARN", args);
        }
    }
}
