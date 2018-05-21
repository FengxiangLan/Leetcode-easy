import math
# Two Sum
# Given an array of integers, return indices of the two numbers such that they add up to a specific target.

# You may assume that each input would have exactly one solution, and you may not use the same element twice.

# Example:
# Given nums = [2, 7, 11, 15], target = 9,

# Because nums[0] + nums[1] = 2 + 7 = 9,
# return [0, 1].
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    sum = 0
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            sum = nums[i] + nums[j]
            if sum == target:
                return [i,j]
   
def reverseInt(x):
        temp = abs(x)
        n = len(str(temp))
        sign = 1
        if temp == x:
            sign = 1
        else:
            sign = -1
        x_list = list(str(temp))
        rever_list = [0 for i in range(n)]
        for i in range(n):
            rever_list[n-1-i] = x_list[i]
        x_rever = 0
        for i in rever_list:
            i = int(i)
            x_rever = x_rever + i * pow(10, (n-1))
            n = n-1
        x_rever = x_rever * sign
        if abs(x_rever) > pow(2, 31):
            x_rever = 0
        return x_rever

def isPalindrome(x):
    if x < 0:
        return False
    else:
        x = list(str(x))
        for i in range(len(x)):
            if x[i] != x[len(x)-1-i]:
                return False
        return True

def romanToInt(s):
    x = 0
    s = list(s)
    d = {'I': 1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    for i in range(0,len(s)-1):
        if d[s[i]] > d[s[i+1]]:
            x = x + d[s[i]]
        if d[s[i]] < d[s[i+1]]:
            if s[i] == 'I' or s[i] == 'X' or s[i] == 'C':
                x = x - d[s[i]]
            else:
                x = x + d[s[i]]
        if d[s[i]] == d[s[i+1]]:
            x = x + d[s[i]]
    x = x + d[s[-1]] 
    return x

def longestCommonPrefix(strs):
    if strs == []:
        return ''

    prefix = strs[0]

    for i in range(1, len(strs)):
        if not prefix:
            return ''
        else:
            while prefix not in strs[i][:len(prefix)] and len(prefix)>0:
                prefix = prefix[:len(prefix)-1]
    return prefix

def isVaild(s):
    stack = []
    for i in range(len(s)):
        if s[i] == '(' or s[i] == '[' or s[i] == '{':
            stack.append(s[i])
        if s[i] == ')':
            if stack == [] or stack.pop() != '(':
                return False
        if s[i] == ']':
            if stack == [] or stack.pop() != '[':
                return False
        if s[i] == '}':
            if stack == [] or stack.pop() != '{':
                return False
    if stack:
        return False
    else:
        return True

def mergeTwoLists(l1, l2):
    head = tial = ListNode(-1)
    while l1 and l2:
        if l1.val < l2.val:
            head.next = l1
            l1 = l1.next
        else:
            head.next =l2
            l2 = l2.next
        head = head.next

    if l1:
        head.next = l1
    if l2:
        head.next = l2
    return tial.next

def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) <= 1:
        return len(nums)
    else:
        length = 0
        for i in range(1, len(nums)):
            if nums[length] != nums[i]:
                length = length + 1
                nums[length] = nums[i]
        return length + 1

def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    if len(nums) == 1:
        if nums[0] == val:
            nums.pop()
            return None
        if nums[0] != val:
            return 1
    elif len(nums) == 0:
        return len(nums)
    else:
        length = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[length] = nums[i]
                length = length + 1
        return length

def strStr(haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    if len(haystack) == len(needle):
        if haystack == needle:
            return 0
        else:
            return -1
    else:
        haystack = list(haystack)
        needle = list(needle)
        for i in range(len(haystack)):
            k = i
            j = 0
            while k < len(haystack) and j < len(needle) and haystack[k] == needle[j]:
                j = j+1
                k = k+1
            if j == len(needle):
                return i
        return -1

def searchInsert(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    first = 0;last = len(nums) - 1
    while first < last:
        mid = (first + last + 1) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            first = mid + 1
        else:
            last = mid - 1
    if nums[last] < target:
        return last + 1
    if target <= nums[last]:
        return last
    if target < nums[first]:
        return first
    return first + 1

def countAndSay(self, n):
    """
    :type n: int
    :rtype: str
    """
    if n == 1:
        return '1'
    last = self.countAndSay(n-1)
    i = 0
    char = last[0]
    result = ''
    for j in range(1,len(last)):
        if last[j] != char:
            result = result + str(j-i) + char
            i = j
            char = last[j]
    result += str(len(last)-i) + char
    return result        

def lengthOfLastWord(s):
    """
    :type s: str
    :rtype: int
    """
    if len(s) == 0:
        return 0
    s = s.split()
    if len(s) > 0:
        return len(s[-1])
    return 0

def plusOne(digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    if len(digits) == 1:
        if digits[0] != 9:
            digits[0] = digits[0] + 1
            return digits
        if digits[0] == 9:
            return [1, 0]
    else:
        integers = 0
        for i in range(len(digits)):
            integers += digits[i] * pow(10, len(digits)-1-i)
        integers = integers + 1
        integers = list(str(integers))
        digits = [0 for i in range(len(integers))]
        for i in range(len(integers)):
            digits[i] = int(integers[i])
        return digits

def addBinary(a, b):
    a = list(a)
    b = list(b)
    int_a = 0
    int_b = 0
    new_int = 0
    for i in range(len(a)):
        int_a = int_a + int(a[i]) * pow(2, len(a)-1-i)
    for j in range(len(b)):
        int_b = int_b + int(b[j]) * pow(2, len(b)-1-j)
    new_int = int_a + int_b
    #print(new_int)
    new_binary = ''
    if new_int == 0:
        new_binary = '0'
        return new_binary
    else:
        d = new_int//2
        while new_int != 0:
            d = new_int//2
            res = new_int % 2
            new_binary = new_binary + str(res)
            new_int = d
        new_binary = new_binary[::-1]   
        return new_binary
def mySqrt(x):
    lo = 0
    hi = x
    while lo <= hi:
        mid = (hi + lo) // 2
        v = mid * mid
        if v < x:
            lo = mid + 1
        elif v > x:
            hi = mid - 1
        else:
            return mid
    return hi
def climbStaurs(n):
    pair = n//2
    if pair == 0:
        return 1
    else:
        ways = 1
        for i in range(1, pair+1):
            print(i)
            temp = 1
            fact = 1
            for j in range(i):
                temp = temp * (n-i-j)
                fact = fact * (j+1)
            print(temp)
            print(fact)
            ways = ways + (temp) // fact
        return ways

def deleteDuplicates(head):
    temp = ListNode(None)
    temp.next = head
    p = temp
    while p and p.next:
        if p.val == p.next.val:
            p.next = p.next.next
        else:
            p = p.next
    return temp.next
                 
def merge(nums1, m, nums2, n):
    nums1[m:] = nums2[:n]
    nums1.sort()


def isSameTree(self, p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    if not p or not q:
        return p == q
    return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# 101
def isSymmetric(self, root):
    if not root:
        return True
    else:
        return self.isMirror(root.left, root.right)

def isMirror(self, left, right):
    if not left and not right:
        return True
    if not left or not right:
        return False

    if left.val == right.val:
        outPAIR = self.isMirror(left.left, right.right)
        inPAIR = self.isMirror(left.right, right.left)
        return outPAIR and inPAIR
    else:
        return False

# 104
def maxDepth(self, root):
    if not root:
        return 0 
    else:
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

# 107
from collections import deque
def levelOrderBottom(self, root):
    if not root:
        return []
    results = [[root.val]]
    queues = deque([root])
    while queues:
        result = []
        for i in range(len(queues)):
            root = queues.popleft()
            if root.left:
                result.append(root.left.val)
                queues.append(root.left)
            if root.right:
                result.append(root.right.val)
                queues.append(root.right)
        if result:
            results.append(result)
    return results[::-1]

#108
def sortedArrayToBST(self,nums):
    if nums:
        mid = len(nums)//2
        midNode_val = nums[mid]
        root = TreeNode(midNode_val)
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

# 110
def isBalanced(self,root):
    if not root:
        return True
    else:
        return self.getHeight(root) != 0
def getHeight(self,root):
    if not root:
        return 1
    left = self.getHeight(root.left)
    right = self.getHeight(root.right)
    if left == 0 or right == 0 or abs(left-right)>1:
        return 0
    return 1 + max(left,right)

# 111
def minDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0
    left = self.minDepth(root.left)
    right = self.minDepth(root.right)
    if not left and not right:
        return 1
    elif not right and left:
        return left + 1
    elif not left and right:
        return right + 1
    else:
        return min(left, right) + 1

#112
def hasPathSum(self, root, sum):
    if not root:
        return False
    if not root.left and not root.right:
        if root.val == sum:
            return True
        else:
            return False
    sum = sum - root.val

    return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right,sum)

# 118
def generate(self, numRows):
    if numRows == 0:
        return []
    if numRows == 1:
        return [[1]]

    results = [[0 for j in range(i+1)] for i in range(0,numRows)]
    results[0] = [1]
    results[1] = [1,1]
    for i in range(2, numRows):
        results[i][0] = 1
        results[i][i] = 1
        for j in range(1, i):
            results[i][j] = result[i-1][j-1] + result[i-1][j]
    return results

# 119
def getRow(rowIndex):
    rowIndex = rowIndex + 1
    results = [1 for i in range(0, rowIndex)]

    i_facts = [1 for i in range(0, rowIndex)]
    for i in range(1,rowIndex):
        i_facts[i] = i_facts[i-1] * i
    for i in range(0, rowIndex):
        results[i] = i_facts[-1]//(i_facts[i] * i_facts[rowIndex-1-i])
    return results

# 121
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return 0
    ans = 0
    pre = prices[0]
    for i in range(1, len(prices)):
        pre = min(pre, prices[i])
        ans = max(prices[i]-pre, ans)
    return ans

# 122
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    ans = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            ans += prices[i] - prices[i - 1]
    return ans
# 125
def isPalindrome(self,s):
    if len(s) <=0:
        return True
    else:
        start = 0
        end = len(s) - 1
        while start < end:
            if not s[start].isalnum():
                start += 1
                continue
            if not s[end].isalnum():
                end -= 1
                continue
            if s[start].lower() != s[end].lower():
                return False
            start +=1
            end -=1
        return True
# 136
def singleNumber(self, nums):
    dic = {}
    for num in nums:
        if num not in dic.keys():
            dic[num] = 1
        else:
            dic[num] = dic[num] + 1
    for key, val in dic.items():
        if val == 1:
            return key

# 141
def hasCycle(self, head):
    fast = head
    slow = head
    while slow and slow.next:
        slow = slow.next.next
        fast = fast.next
        if fast == slow:
            return True
    return False

# 155
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if not self.stack:
            self.stack.append((x,x))
        else:
            self.stack.append((x, min(x, self.stack[-1][-1])))
        

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]

    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
# 157
def getIntersectionNode(self, headA, headB):
    pointA = headA
    pointB = headB
    lenA = 0
    lenB = 0

    while pointA is not None:
        lenA += 1
        pointA = pointA.next

    while pointB is not None:
        lenB += 1
        pointB = pointB.next

    pointA = headA
    pointB = headB

    if lenA > lenB:
        for i in range(lenA-lenB):
            pointA = pointA.next

    if lenA < lenB:
        for i in range(lenB-lenA):
            pointB = pointB.next

    while pointB!=pointA:
        pointA = pointA.next
        pointB = pointB.next

    return pointA

def twoSum(self, numbers, target):
    start = 0
    end = len(numbers) - 1

    while start < end:
        Sum = numbers[start] + numbers[end] 
        if Sum < target:
            start = start + 1
        elif Sum > target:
            end = end - 1
        else:
            return [start + 1, end + 1]

def converToTitle(self, n):
    result = ''
    distance = ord("A")

    while n > 0:
        y = (n-1)%26
        n = (n-1)//26
        s = chr(y+distance)
        result = ''.join((s, result))

    return result

def majorityElement(self,nums):
    dicts = {}
    for num in nums:
        if num not in dicts:
            dicts[num] = 1
        else:
            dicts[num] += 1

    for key in dicts.keys():
        if dicts[key] > len(nums)//2:
            return key

def titleToNumber(self, s):
    numList = []
    for char in s:
        numList.append(ord(char) - ord("A") + 1)

    sum = 0
    l = len(numList)
    for i, n in enumerate(numList):
        sum += n * int(math.pow(26, l-1-i))
    return sum

def trailingZeroes(self, n):
    count, k = 0, 5
    while n:
        k = n // 5
        count += k
        n = k
    return count

def rotate(self, nums, k):
    n = len(nums)
    k = k % n
    nums[:] = nums[n-k:] + nums[:n-k]

# def reverseBits(n):


def hammingWeight(self, n):
    """
    :type n: int
    :rtype: int
    """
    return bin(n).count('1')


def rob(self, nums):
    """
    f(0) = nums[0]
    f(1) = max(num[0], num[1])
    f(k) = max( f(k-2) + nums[k], f(k-1) )
    """
    last, now = 0, 0
        
    for i in nums: last, now = now, max(last + i, now)
            
    return now



def isHappy(self, n):
    record = {}
    while n != 1:
        n = sum([int(i) ** 2 for i in str(n)])
        if n in record:
            return False
        else:
            record[n] = 1
    else:
        return True



def removeElements(self,head, val):
    result = ListNode(None)
    result.next = head
    temp = result
    while temp.next:
        if temp.next.val == val:
            temp.next = temp.next.next
        else:
            temp = temp.next
    return result.next

#204
def countPrimes(self, n):
    if n < 2:
        return 0
    
    dp = [1] * n
    dp[0] = 0
    dp[1] = 0

    for i in range(2, n):
        if dp[i] == 1:
            k = i * i
            if k >= n:
                pass
            while k < n:
                dp[k] = 0
                k = k + i
    return sum(dp)

#205
def isIsomorphic(self, s, t):
    return len(set(s)) == len(set(t)) == len(set(zip(s, t)))

# 206
def reverseList(head):
    result = None
    while head:
        cur = head
        head = head.next
        cur.next = result
        result = cur
    return result 

#217
def containsDuplicate(self, nums):
    return len(nums) != len(set(nums)) 

#219
def containsNearDuplicate(nums, k):
    dic = {}
    for i in range(len(nums)):
        if nums[i] not in dic:
            dic[nums[i]] = i
        else:
            if (i - dic[nums[i]]) <= k:
                return True
            else:
                dic[nums[i]] = i

    return False

# 225 
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queues = []

        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

        

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.stack.pop(-1)
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.stack[-1]
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self.stack == []
        


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

# 226
def invertTree(self, root):
    if not root:
        return None
    left = root.left
    right = root.right
    root.left = self.invertTree(right)
    root.right = self.invertTree(left)
    return root
        
# 231
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n == 0:
        return False
    if n == 1:
        return True
    while n != 1:
        res = n % 2
        if res != 0:
            return False
        n = n // 2
    return True
# 232
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = deque([])
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.stack.popleft()
        

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.stack[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not self.stack 

# 234
def isPalindrome(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    def reverseList(head):

        result = None
        while head:
            cur = head
            head = head.next
            cur.next = result
            result = cur
        return result
        
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
    newHead = reverseList(slow)
    p1 = head
    p2 = newHead
    while p1 and p2:
        if p1.val != p2.val:
            return False
        p1 = p1.next
        p2 = p2.next
    return True

# 235
def lowestCommonAncestor(self, root, p, q):
    maximum = max(p.val, q.val)
    minimum = min(p.val, q.val)
    while not minimum <= root.val <= maximum:
        if minimum > root.val:
            root = root.right
        else:
            root = root.left
    return root


# 237
def deletNode(self, node):
    node.val = node.next.val
    node.next = node.next.next

# 242
def isAnagram(self, s, t):
    return sorted(list(s)) == sorted(list(t))

# 257
def binaryTreePaths(self, root):
    if not root:
        return []
    result = []
    self.dfs(root, "", result)
    return result

def dfs(self, root, path, result):
        if not root.left and not root.right:
            result.append(path + str(root.val))
        if root.left:
            self.dfs(root.left, path + str(root.val) + "->", result)
        if root.right:
            self.dfs(root.right, path + str(root.val) + "->", result)

# 258
def addDigits(self, num):
    """
    :type num: int
    :rtype: int
    """
    if num < 10:
        return num
    return 1 + (num - 1) % 9

# 263
def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return False
    while num%2 == 0 or num%3 == 0 or num%5 == 0:
        if num%2 == 0:
            num = num//2
        if num%3 == 0:
            num = num//3
        if num%5 == 0:
            num = num//5
        
    if num == 1:
        return True
    return False
# 268
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    return (n*(n+1))/2 - sum(nums)

# 278
def firstBadVersion(self, n):
    fBV = 1
    end = n
    while fBV < end:
        mid = fBV + (end - fBV) //2
        if isBadVersion(mid):
            end = mid
        else:
            fBV = mid + 1
    return fBV
#283
def moveZeroes(self, nums):
    i = 0
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            """
            如果这么写意味着赋值出一个tuple，相当于交换nums里 i 和 j 的数值
            如果分开写，那就是先把nums[j] 的值换成了nums[i] 的值，这时候nums j 的值已经变化，然后所以再给nums[i] 赋值 就没有意义了，相当于nums[i] 没有变化
            """
            nums[j],nums[i] = nums[i], nums[j]
            j = j + 1

# 290
"""
a = [1,2,1,3]
b = [5,6,5,6]
c = list(zip(a,b))
"""
def wordPattern(self, pattern, str):
    """
    :type pattern: str
    :type str: str
    :rtype: bool
    """
    str = str.splite()
    compare = set(zip(str, pattern))
    return len(str) == len(pattern) and len(compare) == len(set(str)) == len(set(pattern)) 

# 292
def canWinNim(self, n):
    return not n%4 == 0

# 303
class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.obj = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.obj[i+1] = self.obj[i] + nums[i]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.obj[j+1]-self.obj[i]
            
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)

# 326
def isPowerOfThree(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n > 0:
        return (1162261467%n) == 0
    else:
        return False

# 342
def isPowerOfFour(self, num):
    """
    :type num: int
    :rtype: bool
    """
    return num & (num - 1) == 0 and (num-1)%3 == 0
# 344
def reverseString(self, s):
    return s[::-1]


# 345
def reverseVowels(self,s):
    dic = {"a":1, "e":1, "i":1, "o":1, "u":1, "A":1, "E":1, "I":1, "O":1, "U":1}
    s = list(s)
    first = 0
    end = len(s) - 1
    while first < end:
        if s[first] not in dic:
            first = first + 1
        elif s[end] not in dic:
            end = end - 1
        else:
            s[first], s[end] = s[end], s[first]
            first = first + 1
            end = end - 1
    return "".join(s) 

# 349
def intersection(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    intersection = list(set(nums1).intersection(set(nums2)))
    return intersection

# 350
def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        dict1 = {}
        dict2 = {}
        for i in range(len(nums1)):
            if nums1[i] not in dict1:
                dict1[nums1[i]] = 1
            else:
                dict1[nums1[i]] += 1
        
        for i in range(len(nums2)):
            if nums2[i] not in dict2:
                dict2[nums2[i]] = 1
            else:
                dict2[nums2[i]] += 1
                
        result = []
        for key, val in dict1.items():
            if key in dict2.keys():
                result = result + [key]*min(dict1[key], dict2[key])
            else:
                pass
        return result

# 367
def isPerfectSquare(num):
    if num == 1:
        return True
    if num == 2:
        return False

    first = 1
    end = num
    while first < end:
        mid = (end + first)//2
        if num == mid * mid:
            return True
        elif num < mid * mid:
            end = mid
        else:
            first = mid + 1 
    return False

# 371
def getSum(self, a, b):
    """
    :type a: int
    :type b: int
    :rtype: int
    """
    for _ in range(32):
        a, b = a^b, (a&b)<<1
    return a if a & 0x80000000 else a & 0xffffffff
# 374
def guessNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    first = 1
    end = n
    while first < end:
        mid = first + (end - first)/2
        g = guess(mid)
        if g == -1:
            end = mid - 1
        elif g == 1:
            first = mid + 1
        else:
            return mid
    return first
# 383
def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    dict1 = {}
    for r in ransomNote:
        if r not in dict1:
            dict1[r] = 1
        else:
            dict1[r] += 1
    dict2 = {}
    for m in magazine:
        if m not in dict2:
            dict2[m] = 1
        else:
            dict2[m] += 1
    
    for key in dict1.keys():
        if key not in dict2.keys() or dict1[key] > dict2[key]:
            return False
    return True
# 387
def firstUniqChar(self, s):
    """
    :type s: str
    :rtype: int
    """
    letters='abcdefghijklmnopqrstuvwxyz'
    index=[s.index(l) for l in letters if s.count(l) == 1]
    return min(index) if len(index) > 0 else -1

# 389
# def findTheDifference(self, s, t):
#         """
#         :type s: str
#         :type t: str
#         :rtype: str
#         """
#     return([i for i in t if i not in s or s.count(i)!=t.count(i)][0])
# 400
def findNthDigit(self,n):
    """
    :type n: int
    :rtype: int
    """
    start, size, step = 1, 1, 9
    while n > size * step:
        n -= size * step
        size += 1
        start *= 10
        step *= 10
    return int(str(start + (n-1)//size)[(n - 1) % size])

# 401
def readBinaryWatch(self, num):
    """
    :type num: int
    :rtype: List[str]
    """
    ans = []
    for i in range(0, 12):
        for j in range(0, 60):
            if (bin(i) + bin(j)).count("1") == num:
                ans.append("%d:%02d" % (i, j))
    return ans
# 404
def sumOfLeftLeaves(self, root):
    if not root:
        return 0
    if root.left and not root.left.left and not root.left.right:
        return root.left.val + self.sumOfLeftLeaves(root.right)
    return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)
# 405
def toHex(num):
    d = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f"}
    result = ""
    if num == 0:
        return "0"
    if num < 0:
        num = num + 2**32
    while num != 0:
        res = num % 16
        result = result + d[res]
        num = num // 16

    return result[::-1]

# 409
def longestPalindrome(self, s):
    maxLen = 0
    single = False
    dic = {}
    for c in s:
        if c not in dic:
            dic[c] = 1
        else:
            dic[c] += 1

    for key in dic:
        if dic[key] >= 2:
            count = dic[key]
            res = dic[key] % 2
            d[key] = res
            maxLen += count - res
        if not single:
            if d[key] == 1:
                maxLen += 1
                single = True
    return maxLen
# 412
def fizzBuzz(self,n):
    result = []
    for i in range(1,n+1):
        if i % 3 == 0 and i % 5 != 0:
            result.append("Fizz")
        elif i % 3 != 0 and i % 5 == 0:
            result.append("Buzz")
        elif i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        else:
            result.append(str(i))
    return result
# 414
def thirdMax(self, nums):
    nums = set(nums)
    if len(nums) < 3:
        return max(nums)
    nums.remove(max(nums))
    nums.remove(max(nums))
    return max(nums)
# 415
def addStrings(num1, num2):
    carry = 0
    i = len(num1) - 1
    j = len(num2) - 1
    ans = ""
    for k in reversed(range(0, max(len(num1), len(num2)))):
            a = int(num1[i]) if i >= 0 else 0
            b = int(num2[j]) if j >= 0 else 0
            i, j = i - 1, j - 1
            c = carry
            carry = 0
            sum = a + b + c
            if sum >= 10:
                carry = 1
                ans += str(sum - 10)
            else:
                ans += str(sum)
    if carry == 1:
        ans += "1"
    return ans[::-1]

# 434
def countSegments(self, s):
    """
    :type s: str
    :rtype: int
    """
    return len(s.split())

# 437
def pathSum(self, root, sum):
    self.count = 0
    self.sum = sum

    if not root:
        return 0
    self.dfs(root, [])
    return self.count

def dfs(self, node, vl):
    if not node:
        return 
    vl = [i+node.val for i in vl] + [node.val]
    self.count += vl.count(self.sum)
    self.dfs(node.left, vl)
    self.dfs(node.right, vl)
    return

# 438
from collections import Counter
def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        sCounter = Counter(s[:len(p)-1])
        pCounter = Counter(p)
        ans = []

        for i in range(len(p)-1, len(s)):
            sCounter[s[i]] += 1
            if sCounter == pCounter:
                ans.append(i-len(p)+1)
            sCounter[s[i-len(p)+1]] -= 1
            if sCounter[s[i-len(p)+1]] == 0:
                del sCounter[s[i-len(p)+1]]
        return ans




# 441
def arrangeCoins(self,n):
    return int((((1 + 8*n)**0.5) - 1) / 2)

# 443
#  def compress(self, chars):

# 447
def numberOfBoomerangs(self, points):
    """
    :type points: List[List[int]]
    :rtype: int
    """
    # idea:
    # we compute the distance starting from any given point and we use a hashtable to count the number of the same distance obtained
    # once we finish counting distance for one point, we calculate the combinations = 1 * C^1_N * C^1_(N-1)
    ans = 0
    for p1 in points:
        d = {}
        for p2 in points:
            if p1 != p2:
                dist = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                d[dist] = d.get(dist, 0) + 1
        for k in d:
            ans += d[k] * (d[k] - 1)
    return ans

# 448 
def findDisappearedNumbers(nums):

    return list(set(range(1, len(nums)+1)) - set(nums))

# 453
def minMoves(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    return sum(nums) - len(nums) * min(nums)

# 455
def findContentChildren(self, children, cookies):
    children.sort()
    cookies.sort()

    i = 0
    for cookie in cookies:
        if i >= len(children):
            break
        if children[i] <= cookie:
            i += 1
    return i

# 459
def repeatSubstringPattern(self, s):
    for i in range(0, len(s)//2):
        if len(s) % (i+1) == 0 and s[:i+1] * (len(s)//(i+1)) == s:
            return True
    return False
# 461
def hammingDistance(self, x, y):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    x = x ^ y
    y = 0
    while x:
        y += 1
        x = x & (x - 1)
    return y

# 463
def islandPerimeter(self, grid):

    ans = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            ans += countOfEach(grid, i, j)
    return ans


    def countOfEach(grid, i, j):
        res = 0
        if grid[i][j] == 0:
            return 0
        if i == 0 or i - 1 >= 0 and grid[i-1][j] == 0:
            res += 1
        if i == len(grid) - 1 or i + 1 < len(grid) and grid[i+1][j] == 0:
            res += 1
        if j == 0 or j - 1 >= 0 and grid[i][j-1] == 0:
            res += 1
        if j == len(grid[0]) - 1 or j + 1 < len(grid[0]) and grid[i][j+1] == 0:
            res += 1
        return res


# 475
def findRadius(self, houses, heaters):
    """
    :type houses: List[int]
    :type heaters: List[int]
    :rtype: int
    """
    houses.sort()
    heaters.sort()
    heaters=[float('-inf')]+heaters+[float('inf')] # add 2 fake heaters
    ans,i = 0,0
    for house in houses:
        while house > heaters[i+1]:  # search to put house between heaters
            i +=1
        dis = min (house - heaters[i], heaters[i+1]- house)
        ans = max(ans, dis)
    return ans
# 476
def findComplement(self, num):
    return ~num & ((1<<num.bit_length())-1)
 
# 482
def licenseKeyFormatting(self, S, K):
    s = S.split("-")
    s = "".join(s)
    n = len(s)
    start = n % K
    res = []
    if start != 0:
        res.append(s[:start].upper())
    
    for k in range(0, (len(s) - start) / K):
        res.append(s[start:start+K].upper())
        start += K
    return "-".join(res)

# 485
def findMaxConsecutiveOnes(self, nums):
    ans = 0
    count = 0
    for num in nums:
        if num == 1:
            count += 1
        else:
            count = 0
        ans = max(ans, count)
    return ans
# 492
def constructRectangle(self, area):
    """
    :type area: int
    :rtype: List[int]
    """
    root = int(area ** 0.5)
    while root > 0:
        if area % root == 0:
            return int(area / root), root
        root -= 1
# 496
def nextGreaterElement(self, findNums, nums):
    ans = [-1]*len(findNums)
    d = {}

    for i in range(len(findNums)):
        if findNums[i] not in d:
            d[findNums[i]] = i

    temp = []
    for num in nums:
        while temp and temp[-1] < num:
            top = temp.pop()
            if top in d:
                ans[d[top]] = num
        temp.append(num)
    return ans

# 500
def findWords(self, words):
    ans = []
    d = {}
    row1 = "qwertyuiop"
    row2 = "asdfghjkl"
    row3 = "zxcvbnm"
    for r in row1:
        d[r] = 1
    for r in row2:
        d[r] = 2
    for r in row3:
        d[r] = 3

    for word in words:
        same = True
        pre = d[word[0].lower]
        for c in word:
            if d[c.lower()] != pre:
                same = False
                break
        if same:
            ans.append(word)
    return ans 
# 501
def helper(self, root, cache):
    if root == None:
        return
    if root.val not in cache:
        cache[root.val] = 1
    else:
        cache[root.val] += 1
    self.helper(root.left, cache)
    self.helper(root.right, cache)
    return
def findMode(self, root):
    if root == None:
        return []
    cache = {}
    self.helper(root, cache)
    max_freq = max(cache.values())
    result = [k for k, v in cache.items() if v == max_freq]
    return result
# 504
def convertToBase7(num):
    tag = 1
    if num < 0:
        tag = -1

    num = abs(num)
    result = ""
    if num == 0:
        result = "0"
    while num != 0:
        factor = num // 7
        res = num % 7
        result = result + str(res)
        num = factor

    if tag == 1:
        return result[::-1]
    else:

        return "-" + result[::-1]

# 506
def findRelativeRanks(self, nums):
    ans = [""] * len(nums)
    scores = []
    for i, num in enumerate(nums):
        scores.append((num, i))
    scores.sort(reverse=True)
    rankTitles = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    rank = 0
    for _, i in scores:
        if rank > 2:
            ans[i] = str(rank + 1)
        else:
            ans[i] = rankTitles[rank]
        rank += 1
    return ans

# 507
def checkPerfectNumber(self, num):
    """
    :type num: int
    :rtype: bool
    """
    ans = 1
    div = 2
    while div ** 2 <= num:
        if num % div == 0:
            ans += div
            ans += num / div
        div += 1
    return ans == num if num != 1 else False
# 520
def detectCapitalUse(self, word):
    if word == word.upper() or word == word.lower():
        return True
    else:
        if word[0] == word[0].upper() and word[1:] == word[1:].lower():
            return True
        else:
            return False
# 521
def findLUSlength(self, a, b):
    return max(len(a), len(b)) if a != b else -1
# 530
def getMinimumDifference(self, root):
    self.val = None
    self.ans = float("inf")
    def inorder(root):
        if not root:
            return
        inorder(root.left)
        if self.val is not None:
            self.ans = min(self.ans, abs(root.val - self.val))
        self.val = root.val
        inorder(root.right)

    inorder(root)
    return self.ans
# 532
def findPairs(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    res = 0
    c = collections.Counter(nums)
    for i in c:
        if k > 0 and i + k in c or k == 0 and c[i] > 1:
            res += 1
    return res

# 538
def convertBST(self, root):
    self.sum = 0

    def greaterTree(node):
        if not node:
            return None
        right = greaterTree(node.right)
        self.sum += node.val
        new_node = TreeNode(self.sum)
        new_node.right = right
        new_node.left = greaterTree(node.left)
        return new_node

    return greaterTree(root)

    
# 541
def reverseStr(s, k):
    ans = ""
    while s:
        temp = s[:k]
        ans = ans + temp[::-1] + s[k:2*k]
        s = s[2*k:]
    return ans

# 543
def diameterOfBinaryTree(self, root):
    self.ans = 0

    def DFS(root):
        if not root:
            return 0
        left = DFS(root.left)
        right = DFS(root.right)
        self.ans = max(self.ans, left + right)
        return max(left, right) + 1

    DFS(root)
    return self.ans

# 551
def checkRecord(self, s):
    if s.count("A") >= 2 or "LLL" in s:
        return False
    return True

# 557
def reverseWords(s):
    sList = s.split()
    ans = []
    for strr in sList:
        ans.append(strr[::-1])

    return " ".join(ans)

# 561
def arrayPairSum(nums):
    ans = 0
    nums.sort()
    i = 0
    while i < (len(nums)-1):
        print(nums[i])
        ans = ans + min(nums[i], nums[i+1])
        i = i+2

    return ans

# 563
def findTilt(self, root):
    self.ans = 0
    def DFS(root):
        if not root:
            return 0
        left = DFS(root.left)
        right = DFS(root.right)
        self.ans += abs(left - right)
        return root.val + left + right

    DFS(root)
    return self.ans

# 566
def matrixReshape(self, nums, r, c):
    if r * c != len(nums) * len(nums[0]):
        return nums

    m = len(nums)
    n = len(nums[0])
    ans = [[0] * c for _ in range(r)]
    for i in range(r*c):
        ans[i//c][i%c] = nums[i//n][i%n]
    return ans
# 572
def isMatch(self, s, t): 
    if not (s and t):
        return s is t
    return (s.val == t.val and self.isMatch(s.left, t.left) and self.isMatch(s.right, t.right))
def isSubtree(self, s, t):
    if self.isMatch(s,t):
        return True
    if not s:
        return False
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

# 575
def distributeCandies(self, candies):
    """
    :type candies: List[int]
    :rtype: int
    """
    temp = set(candies)
    if 2*len(temp) >= len(candies):
        return len(candies)//2
    else:
        return len(temp)

# 581
def findUnsortedSubarray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    
    is_same = [a == b for a, b in zip(nums, sorted(nums))]
    return 0 if all(is_same) else len(nums) - is_same.index(False) - is_same[::-1].index(False)

# 594
from collections import Counter
def findLHS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    ans = 0
    d = Counter(nums)
    for num in nums:
        if num + 1 in d:
            ans = max(ans, d[num] + d[num+1])
    return ans 

# 598
from functools import reduce
def maxCount(self, m, n, ops): 
    return reduce(operator.mul, map(min, zip(*ops + [[m,n]])))
# 599
def findRestaurant(list1, list2):
    d = {}
    for i,word in enumerate(list1):
        d[word] = i

    ansL = {}
    for j,word in enumerate(list2):
        if word in d:
            ansL[word] = d[word] + j

    if len(ansL) == 0:
        return []

    else:
        minVal = min(list(ansL.values()))
        ans = []
        for key, val in ansL.items():
            if val == minVal:
                ans.append(key)
        return ans

# 605
def canPlaceFlowers(flowerbed, n):
    """
    :type flowerbed: List[int]
    :type n: int
    :rtype: bool
    """
    ans = 0
    cnt = 1
    for plot in flowerbed:
        if plot == 0:
            cnt += 1
            #print(cnt)
        else:
            ans += abs(cnt - 1) // 2
            cnt = 0
            #print(ans)
    return ans + cnt // 2 >= n

# 606
def tree2str(self, t):
    def preorder(root):
        if root is None:
            return ""
        s = str(root.val)
        l = preorder(root.left)
        r = preorder(root.right)
        if r == "" and l == "":
            return s
        elif l == "":
            s += "()"+ "("+r+")"
        elif r == "":
            s += "("+l+")"
        else:
            s += "("+l+")" + "(" + r + ")"
        return s
    return preorder(t)
# 617
def mergeTrees(self, t1, t2):
    if t1 or t2:
        root = TreeNode((t1 and t1.val or 0) + (t2 and t2.val or 0))
        root.left = self.mergeTrees(t1 and t1.left, t2 and t2.left)
        root.right = self.mergeTrees(t1 and t1.right, t2 and t2.right)
        return root


# 628
def maximumProduct(self, nums):
    nums.sort()
    return max(nums[-1] * nums[-2] * nums[-3], nums[0] * nums[1] * nums[-1])
# 633
def judgeSquareSum(self, c):
    n = int(c ** 0.5)
    start = 0
    end = n
    while start <= end:
        mid = start ** 2 + end ** 2
        if mid == c:
            return True
        elif mid < c:
            start += 1
        else:
            end -= 1
    return False
# 637
from collections import deque
def averageOfLevels(self, root):
    ans = []
    queue = deque([root])
    while queue:
        s = 0
        n = len(queue)
        for _ in range(n):
            top = queue.popleft()
            s += top.val
            if top.left:
                queue.append(top.left)
            if top.right:
                queue.append(top.right)
        ans.append(float(s)/n)
    return ans
# 643
def findMaxAverage(self, nums, k):
    subSum = 0
    ans = float("-inf")
    queue = deque([])
    for num in nums:
        queue.append(num)
        subSum += num
        if len(queue) > k:
            subSum -= queue.popleft()
        if len(queue) == k:
            ans = max(ans, float(subSum)/k)
    return ans
def findMaxAverage(self, nums, k):
    maxi = sum(nums[:k])
    temp = maxi
    for i in range(len(nums) - k):
        temp = temp - nums[i] + nums[i + k]
        if temp > maxi:
            maxi = temp
    return float(maxi) / float(k)
# 645
def findErrorNums(self, nums):
    d = Counter(nums)
    n = len(nums)
    ans = [0,0]
    for i in range(1, n+1):
        if (i in d and d[i] == 2):
            ans[0] = i
        if i not in d:
            ans[1] = i
        
    return ans
def findErrorNums(self, nums):
    return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]
# 653
def findTarget(self, root, k):
    if not root:
        return False
    bfs = [root]
    s = set()
    for i in bfs:
        if k - i.val in s: return True
        s.add(i.val)
        if i.left:
            bfs.append(i.left)
        if i.right:
            bfs.append(i.right)
    return False
# 657
def judgeCircle(self, moves):
    x = y = 0
    dirs = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}
    for move in moves:
        dx, dy = dirs[move]
        x += dx
        y += dy
    return x == y == 0
def judgeCircle(self, moves):
    return moves.count('L') == moves.count('R') and moves.count('U') == moves.count('D')
# 661
"""
DON'T UNDERSTAND
"""
def imageSmoother(self, M):
    """
    :type M: List[List[int]]
    :rtype: List[List[int]]
    """
    m = len(M)
    n = len(M[0])
    ans = [[0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            cnt = 0
            sums = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    newi, newj = i + di, j + dj
                    if 0 <= newi < m and 0 <= newj < n:
                        cnt += 1
                        sums += M[newi][newj]
            ans[i][j] = sums // cnt
    return ans
# 665
def checkPossibility(self, nums):
    one, two = nums[:], nums[:]
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            one[i] = nums[i + 1]
            two[i + 1] = nums[i]
            break
    return one == sorted(one) or two == sorted(two)
# 669
def trimBST(self, root, L, R):
    def trim(node):
        if node:
            if node.val > R:
                return trim(node.left)
            elif node.val < L:
                return trim(node.right)
            else:
                node.left = trim(node.left)
                node.right = trim(node.right)
                return node

    return trim(root)

# 671
def findSecondMinimumValue(self, root):
    self.ans = float("inf")
    minV = root.val

    def DFS(node):
        if node:
            if minV < node.val < self.ans:
                self.ans = node.val
            elif node.val == minVal:
                DFS(node.left)
                DFS(node.right)
    DFS(root)
    return self.ans if self.ans < float("inf") else -1

# 674
def findLengthOfLCIS(nums):
    
    if len(nums) == 0:
        return 0
    nums = nums + [float("-inf")]
    ans = deque([nums[0]])
    res = []
    for num in nums[1:]:
        top = ans[-1]
        if num > top:
            ans.append(num)
        else:
            res.append(len(ans))
            ans.clear()
            ans.append(num)
    nums.pop()
    if len(ans) == len(nums):
        return len(nums)
    return sorted(res)[-1]
nums = [1,3,5,4,2,3,4,5]
findLengthOfLCIS(nums)

# 680
def vailPalindrome(self, s):
    left = 0
    right = len(s) - 1
    while left < right:
        if s[left] != s[right]:
            one = s[left:right]
            two = s[left+1:right+1]
            return one == one[::-1] or two == two[::-1]
        left = left + 1
        right = right - 1 
    return True
# 682
def calPoint(self, ops):
    ans = []
    for score in ops:
        if score == "C":
            ans.pop()
        elif score == "D":
            ans.append(ans[-1] * 2)
        elif score == "+":
            ans.append(ans[-1] + ans[-2])
        else:
            ans.append(int(score))
    return sum(ans)

# 686
def repeatedStringMatch(self, A, B):
    C = ""
    for i in range(len(B)/len(A) + 3): 
        if B in C:
            return i
        C += A
    return -1
# 687
def longestUnivaluePath(self, root):
    ans = [0]

    def traverse(node):
        if not node:
            return 0
        left_len = traverse(node.left)
        right_len = traverse(node.right)
        left = (left_len + 1) if node.left and node.left.val == node.val else 0
        right = (right_len + 1) if node.right and node.right.val == node.val else 0
        ans[0] = max(ans[0], left + right)
        return max(left, right)

    traverse(root)
    return ans[0]
# 690
def getImportance(self, employees, id):
    emps = {employee.id: employee for employee in employees}
    def DFS(id):
        sub_imp = sum([DFS(sub_id) for sub_id in emps[id].subordinates])
        return emps[id].importance + sub_imp
    return DFS(id)
# 693
def hasAlternatingBits(self, n):
    return '00' not in bin(n) and '11' not in bin(n) 
# 695
"""
DON'T UNDERSTAND
"""
def maxAreaOfIsland(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if not grid or not grid[0]:
        return 0
    r, c =  len(grid), len(grid[0])
    finished = collections.defaultdict(int)
    res_ = {finished[(i,j)] for i in range(r)
                     for j in range(c) if grid[i][j] == 0}

    def dfs(i,j):
        open_list.append((i,j))
        while open_list:
            i,j = open_list.pop()
            finished[(i,j)]
            close_list.append((i,j))
            temp = [(m,n) for m,n in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)] if m>=0 and m<r and n>=0 and n<c and grid[m][n]]
            if temp:
                 for item in temp:
                    if item not in open_list and item not in close_list:
                        open_list.append(item)
        res_.add(len(close_list))
    for i in range(r):
        for j in range(c):
            if (i,j) not in finished:
                open_list = []
                close_list = []
                dfs(i,j)
    return max(res_)

# 696
def countBinarySubstrings(self, s):
    """
    :type s: str
    :rtype: int
    """
    l = [len(list(v)) for k, v in itertools.groupby(s)]
    return sum(list(map(lambda a, b: min(a, b), l[:-1], l[1:])))

# 697
def findShortestSubArray(self, nums):
    first = {}
    last = {}
    for i, v in enumerate(nums):
        first.setdefault(v,i)
        last[v] = i

    c = Counter(nums)
    degree = max(c.values())

    return min(last[v] - first[v] + 1 for v in c if c[v] == degree)

# 717
def isOneBitCharacter(self, bits):
    if not bits:
        return False

    n = len(bits)
    index = 0
    while index < n:
        if index == n-1:
            return True
        if bits[index] == 1:
            index += 2
        else:
            index += 1
    return False

# 720
"""
NOT UNDERSTAND
"""
def longestWord(self, words):
    length2wordset = {}
    for word in words:
        if len(word) not in length2wordset:
            length2wordset[len(word)] = set()
        length2wordset[len(word)].add(word)
    if 1 not in length2wordset:
        return ''
    length = 2
    while length < 31:
        if length not in length2wordset:
            break
        wordset,updwordset = length2wordset[length],set()
        for word in wordset:
            if word[:-1] in length2wordset[length-1]:
                updwordset.add(word)
        if not updwordset:
            break
        length2wordset[length] = updwordset
        length += 1
    return min(length2wordset[length-1])
    

# 724
def pivotIndex(self, nums):
    left = 0
    right = sum(nums)
    for index, num in enumerate(nums):
        right -= num
        if left == right:
            return index
        left += num
    return -1
# 728
def selfDividingNumber(self, left, right):
    ans = []
    for i in range(left, right+1):
        for j in str(i):
            if j == "0" or i % int(j) != 0:
                break
        else:
            ans.append(i)
    return ans


# 733
def floodFill(self, image, sr, sc, newColor):
	nrow = len(image)
	ncol = len(image[0])
	target = image[sr][sc]

	def DFS(r, c):
		if 0 <= r < nrow and 0 <= c < ncol and image[r][c] == target:
			image[r][c] = newColor
			DFS(r+1, c)
			DFS(r-1, c)
			DFS(r, c+1)
			DFS(r, c-1)


	if target != newColor:
		DFS(sr,sc)

	return image
# 744
def nestGreatestLetter(self, letters, target):
	left = 0
	mid = ((len(letters) - 1)//2)
	right = len(letters)-1
    
	while left <= right:
		if letters[mid] <= target:
			left = mid + 1
		elif letters[mid] > target:
			right = mid - 1
		mid = (left + (right - left) // 2)
	return letters[left] if left < len(letters) else letters[0]

# 746
def minCostClimbingStairs(self, cost):
    dp = [0]*(len(cost))
    dp[0], dp[1]=cost[0], cost[1]
    
    for i in range(2,len(cost)):
        dp[i] = min(dp[i-2]+cost[i], dp[i-1]+cost[i])
    
    return min(dp[-2], dp[-1])
# 747
def dominantIndex(nums):
    if len(nums) == 1:
        return 0
    if len(nums) == 0:
        return -1
    d = {}
    for i,n in enumerate(nums):
        d[n] = i
    tmp = sorted(nums)
    biggest = tmp[-1]
    secondBig = tmp[-2]
    if biggest >= (secondBig*2):
        for key, val in d.items():
            if key == biggest:
                return val
    else:
        return -1

# 762
def countPrimeSetBits(self, L, R):
    ans = []
    for i in range(L, R+1):
        st = str(bin(i)[2:])
        ans.append(st.count("1"))
    k = 0
    for i in ans:
        if i in [2,3,5,7,11,13,17,19]:
            k +=1
    return k

# 766
def isToeplitzMatrix(self, m):
    for i in range(len(m)-1):
        for j in range(len(m[0])-1):
            if m[i][j] != m[i+1][j+1]:
                return False
    return True
# 771
def numJewelsInStones(self, J, S):
    d = {}
    for s in S:
        if s not in d:
            d[s] = 1
        else:
            d[s] += 1

    ans = 0
    for j in J:
        if j in d:
            ans += d[j]

    return ans

# 783
def minDiffInBST(self, root):
    nodes = []
    queue = [root,]
    while queue:
        point = queue.pop(0)
        nodes.append(point.val)
        if point.left:
            queue.append(point.left)
        if point.right:
            queue.append(point.right)
    node.sort()
    result = float("inf")
    for index, value in enumerate(nodes[1::]):
        result = min(value - nodes[index], result)
    return result

# 784
def letterCasePermutation(self, S):
    ans = [S]
    for i in range(len(S)):
        if S[i].isalpha():
            for an in ans:
                s = list(an)
                if s[i].islower():
                    s[i] = s[i].upper()
                else:
                    s[i] = s[i].lower()
                t = "".join(s)
                if t not in ans:
                    ans.append(t)
    return ans

# 788
def rotatedDigits(self, N):
    """
    :type N: int
    :rtype: int
    """
    count=0
    for i in range(1,N+1):
        s =str(i) 
        if '3' in s or '4' in s or'7' in s :
            continue
        elif '2' in s or '5' in s or'6' in s or '9' in s:
            count+=1   
    return count

# 796
def rotateString(self, A, B):
    return len(A) == len(B) and A in (B+B)

# 804
def uniqueMorseRepresentations(self, words):
    """
    :type words: List[str]
    :rtype: int
    """
    d = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
         "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    temp = ''
    dict = {}
    for i in words:
        for j in i:
            temp += d[ord(j)-97]
        dict[temp] = i
        temp = ''
    return len(dict)

# 806
def numberOfLines(self, widths, S):
    widths_dict = dict(zip(string.ascii_lowercase, widths))
    count = 0
    lines = 0
    for s in S:
        count += widths_dict[s]
        if count > 100:
            lines += 1
            count = widths_dict[s]
        elif count == 100:
            lines += 1
            count = 0
    return [lines + 1, count]
# 811
from collections import defaultdict

def subdomainVisits(self, cpdomains):
    count = defaultdict(int)
    for cpdomain in cpdomains:
        c, domain = cpdomain.split()
        c = int(c)
        while "." in domain:
            count[domain] += c
            domain = domain.split('.', 1)[1]
        else:
            count[domain] += c
    return ["%d %s" % (v, k) for k, v in count.items()]
# 812
def largestTriangleArea(self, points):
    """
    :type points: List[List[int]]
    :rtype: float
    """
    max_area = 0
    for i in range(len(points)-2):
        for j in range(i+1, len(points)-1):
            for k in range(j+1, len(points)):
                max_area = max(max_area, self.area(points[i], points[j], points[k]))
    return max_area
                
def area(self, a, b, c):
    a1 = ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
    b1 = ((c[0]-b[0])**2+(c[1]-b[1])**2)**0.5
    c1 = ((a[0]-c[0])**2+(a[1]-c[1])**2)**0.5
    p = (a1+b1+c1)/2
    area = (p*abs(p-a1)*abs(p-b1)*abs(p-c1))**0.5
    return area
# 819
import string
def mostCommonWord(paragraph, banned):
    for c in string.punctuation:
        paragraph =paragraph.replace(c, "")
    paragraph = (paragraph.lower()).split()
    d = Counter(paragraph)
    d = sorted(d.items(), key = lambda x:x[1], reverse = True)
    for key, val in d:
        if key not in banned:
            return key

# 821
def shortestToChar(self, S, C):
    """
    :type S: str
    :type C: str
    :rtype: List[int]
    """
    return [min(abs(i - ll) for ll in [i for i, e in enumerate(S) if e == C]) for i in range(len(S))]
        




