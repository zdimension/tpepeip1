﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace tpepeip1
{
    public class SlidingBuffer<T> : IEnumerable<T>
    {
        private readonly Queue<T> _queue;
        private readonly int _maxCount;

        public SlidingBuffer(int maxCount)
        {
            _maxCount = maxCount;
            _queue = new Queue<T>(maxCount);
        }

        public T this[int pos] => _queue.ToList()[pos];

        public void Add(T item)
        {
            if (_queue.Count == _maxCount)
                _queue.Dequeue();
            _queue.Enqueue(item);
        }

        public void Clear()
        {
            _queue.Clear();
        }

        public int Count => _queue.Count;

        public IEnumerator<T> GetEnumerator()
        {
            return _queue.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
