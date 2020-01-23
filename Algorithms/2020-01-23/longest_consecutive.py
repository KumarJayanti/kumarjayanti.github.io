def longest_cosecutive(nums):
    '''
    Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
    For example, given [100, 4, 200, 1, 3, 2], the longest consecutive element sequence is [1, 2, 3, 4].
    '''
    max_len = 0
    # stores <num> : (left_bound, right_bound)
    bounds = dict()
    for num in nums:
        #if number seen before ignore
        if num in bounds:
            continue
       
        # new number so start with left and right bound set to the number
        left_bound, right_bound = num, num

        # if predecessor in dictionary the expand the left_bound to predecessor
        if num - 1 in bounds:
            left_bound = bounds[num - 1][0]
 
        # if successor in dictionary expand the right_bound 
        if num + 1 in bounds:
            right_bound = bounds[num + 1][1]

        # now store the left-bound and right-bound of num
        bounds[num] = left_bound, right_bound

        # adjust the left and right bounds for the left_bound and right_bound
        bounds[left_bound] = left_bound, right_bound
        bounds[right_bound] = left_bound, right_bound

        max_len = max(right_bound - left_bound + 1, max_len)

    return max_len      

def main():
    args = [100, 4, 200, 1, 3, 2, 5]
    length = longest_cosecutive(args)
    print("longest consecutive chain length=" , length)

if __name__ == "__main__":
    main()

