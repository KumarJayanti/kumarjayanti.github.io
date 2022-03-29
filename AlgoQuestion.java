package com.mycompany;

/*
  Algo question:
  N people, M restaurants, on a one-dimensional street.  All the people are to meet at one restaurant,
  find the restaurant, where total travel distance from all people to that restaurant is the shortest among all restaurants.

  The problem can be solved with brutal force with O(n*m) complexity.  Question is how to solve it faster than brutal force.
  Describe the algorithm, and why it is better than brutal force.  Coding is not required

  [zc] this is one dimensional street.

  A list of people can be represented with their coordinates (p1, p2, ... pn).

  A list of restaurant can be represented with their coordinates (r1, r2, ... rm)

  The distance between i-th people and j-th restaurant can be calculated as ABS(pi - rj)

  Time Complexity of BruteForce Algorithm 1:  O(M * N)

  Time Complexity of Proposed Algorithm 2 : O(M+N)
  This Algorithm tries to compute the distance in two passes of the Combined Sorted List
      Once from Left to Right and Another time from Right to Left
      Time Complexity of the Algorithm :
      O(M+N) : counting sort
      O(M+N) : Left to Right Traversal
      O(M+N) : Right to Left Traversal
      O(M) : traversal over the restautart list to find the minimal cost
      Overall Complexity =  O(3 * (M+N) + M), removing constant multiplier and lower order terms it is  O(M+N)

      Space Complexity :  Needs O(M+N) extra storage for the combine list and O(M) addtional space to store
      two lists of Restaurant objects for Left to Right and Right to Left Traversal respectively.

 */

import java.util.*;

public class AlgoQuestion {

    public static void main(String[] args) {

        int[] people = new int[]{1,2,4,6,7};
        int[] restaurants = new int[]{1,3,5};
        int restaurantWithShortedDist = getShortestDistanceBruteForce(people, restaurants);
        System.out.println("Nearest Restaurant for all people is: " + (restaurantWithShortedDist));

        restaurantWithShortedDist = getShortestTotalDistanceOptimal(people, restaurants);
        System.out.println("Nearest Restaurant for all people is: " + (restaurantWithShortedDist));

        people = new int[]{3,4,2,5,9};
        restaurants = new int[]{4,6,7};
        restaurantWithShortedDist = getShortestDistanceBruteForce(people, restaurants);
        System.out.println("Nearest Restaurant for all people is: " + (restaurantWithShortedDist));

        restaurantWithShortedDist = getShortestTotalDistanceOptimal(people, restaurants);
        System.out.println("Nearest Restaurant for all people is: " + (restaurantWithShortedDist));

        people = new int[]{2,6,8,10,11};
        restaurants = new int[]{7,9};
        restaurantWithShortedDist = getShortestDistanceBruteForce(people, restaurants);
        System.out.println("Nearest Restaurant for all people is: " + (restaurantWithShortedDist));

        restaurantWithShortedDist = getShortestTotalDistanceOptimal(people, restaurants);
        System.out.println("Nearest Restaurant for all people is: " + (restaurantWithShortedDist));

    }

    public static int getShortestDistanceBruteForce(int[] people, int[]  restaurants) {
        int resultRestaurant= -1;
        int globalMin = Integer.MAX_VALUE;
        int totalOperations = 0;
        for (int r=0; r < restaurants.length; r++) {
            int currentRSum = 0;
            for (int p=0; p < people.length; p++) {
                currentRSum += Math.abs(people[p] - restaurants[r]);
                totalOperations++;
                if (currentRSum > globalMin) {
                    break;
                }
            }
            if (currentRSum < globalMin) {
                globalMin = currentRSum;
                resultRestaurant = r;
            }
        }
        System.out.println("Minimum Distance for all people: " + globalMin);
        return resultRestaurant;
    }

    public static  class Triple {
        //distance
        int d;
        //r == true indicates restaurant
        //otherwise person
        boolean r;
        int index;

        public Triple(Integer d, Integer indx, boolean r) {
            this.d = d;
            this.r = r;
            this.index = indx;
        }
        public int getDistance() {
            return d;
        }
    }

    public static class Restaurant {
       int currentCost;
       int cumCost;
       int cumPeopleCount;
       int peopleCount;
       int currentCostRight;

       Triple r;

       public Restaurant(Triple r) {
           this.r = r;
       }

       //assumes setCostAndCount method already called.
        public void setCumCost(Restaurant prev) {
           //if prev is null, it means i am the first restaurant in the list
            if(prev == null) {
                cumCost = 0;
                cumPeopleCount = peopleCount;
                return;
            }
            cumCost = prev.cumCost+ prev.currentCost + Math.abs(r.d - prev.r.d) * prev.cumPeopleCount;
            cumPeopleCount = peopleCount + prev.cumPeopleCount;
        }

        public void setCostAndCount(int c, int count) {
           currentCost = c;
           peopleCount = count;
        }

        public void updateCostAndCount(int costUpdate, int countUpdate) {
            currentCost += costUpdate;
            peopleCount += countUpdate;
            currentCostRight = costUpdate;
        }
    }

    //Linear implementation.
    public static int getShortestTotalDistanceOptimal(int[] people, int[]  restaurants) {

        int[] restaurantDistance = new int[restaurants.length];
        List<Restaurant> restaurantsListLR = new ArrayList<>();
        List<Restaurant> restaurantsListRL = new ArrayList<>();
        List<Triple> combined = new ArrayList<>();
        for (int i=0; i< people.length; i++) {
            combined.add(new Triple(people[i], i, false));
        }
        for (int i=0; i< restaurants.length; i++) {
            Triple t = new Triple(restaurants[i], i, true);
            combined.add(t);
            restaurantsListLR.add(new Restaurant(t));
            restaurantsListRL.add(new Restaurant(t));
        }

        //TODO: do a Counting sort for people with restaurants
        //based on the distance from origin.
        //This will be O(M+N) when changed to Counting Sort
        combined.sort(Comparator.comparing(Triple::getDistance));

        int runningCost = 0;
        int runningCount = 0;

        Restaurant prev = null;
        Restaurant current = null;
        for (int i=0; i < combined.size(); i++) {
            Triple t = combined.get(i);
            if (t.r) {
                //restaurant handling
                current = restaurantsListLR.get(t.index);
                current.setCostAndCount(Math.abs((runningCount * t.d) - runningCost), runningCount);
                runningCost = 0;
                runningCount = 0;
                current.setCumCost(prev);
                prev = current;

            } else {
               //people handling
                runningCost += t.d;
                runningCount++;
            }
        }

        //if at the end of this loop we have runningCost it needs to be attributed
        //to current restaurant
        if (runningCount > 0) {
            current.updateCostAndCount(Math.abs((runningCount * current.r.d) - runningCost), runningCount);
        }

        runningCost = 0;
        runningCount = 0;
        prev = null;
        current = null;

        for (int i= combined.size() -1; i >=0; i--) {
            Triple t = combined.get(i);
            if (t.r) {
                //restaurant handling
                current = restaurantsListRL.get(t.index);
                current.setCostAndCount(Math.abs((runningCount * t.d) - runningCost), runningCount);
                runningCost = 0;
                runningCount = 0;
                current.setCumCost(prev);
                prev = current;

            } else {
                //people handling
                runningCost += t.d;
                runningCount++;
            }
        }

        //if at the end of this loop we have runningCost it needs to be attributed
        //to current restaurant
        if (runningCount > 0) {
            current.updateCostAndCount(Math.abs((runningCount * current.r.d) - runningCost), runningCount);
        }

        int minCost = Integer.MAX_VALUE;
        int minIndex = -1;
        for (int i=0; i < restaurantsListLR.size(); i++) {
            Restaurant rL = restaurantsListLR.get(i);
            Restaurant rR = restaurantsListRL.get(i);
            int totalCost = 0;
            if (i==0) {
                totalCost = rL.currentCost + rR.cumCost + (rR.currentCost - rR.currentCostRight);
            } else if (i==restaurantsListLR.size() -1) {
                totalCost = (rL.currentCost - rL.currentCostRight) + rL.cumCost + rR.currentCost;
            } else {
                totalCost = rL.currentCost + rR.currentCost + rL.cumCost + rR.cumCost;
            }
            if (totalCost < minCost) {
                minCost = totalCost;
                minIndex = i;
            }
            //System.out.println("Total Cost at Restaurant:" + i + "=" + totalCost);
        }

        System.out.println("Minimum Distance for all people: " + minCost);
        return minIndex;
    }

    //Assumption the Street is of fixed length so we can apply Counting Sort
    //for the combined list of People and Restaurants
    public static int getShortestTotalDistance(int[] people, int[]  restaurants) {

        int[] restaurantDistance = new int[restaurants.length];
        List<Triple> combined = new ArrayList<>();
        for (int i=0; i< people.length; i++) {
            combined.add(new Triple(people[i], i, false));
        }
        for (int i=0; i< restaurants.length; i++) {
            combined.add(new Triple(restaurants[i], i, true));
        }

        //TODO: do a Counting sort for people with restaurants
        //based on the distance from origin.
        //This will be O(M+N) when changed to Counting Sort
        combined.sort(Comparator.comparing(Triple::getDistance));

        //This loop is O(M+N)
        int personSumTillNow = 0;
        int personCountTillNow = 0;
        int restaurantSeenTillNow = -1;

        for (int i=0; i < combined.size(); i++) {
            Triple t = combined.get(i);
            if (t.r) {
                //restaurant handling
                restaurantDistance[t.index] = Math.abs(personCountTillNow * t.d - personSumTillNow);
                restaurantSeenTillNow = t.index;
            } else {
                //people handling
                personSumTillNow += t.d;
                personCountTillNow += 1;
                //update only restaurants seen till now
                for (int j=restaurantSeenTillNow; j >=0; j--) {
                    restaurantDistance[j] += (Math.abs(t.d - restaurants[j]));
                }
            }
        }

        int resultRestaurant= -1;
        int globalMin = Integer.MAX_VALUE;
        for (int i=0; i < restaurantDistance.length; i++) {
            if (restaurantDistance[i] < globalMin) {
                globalMin = restaurantDistance[i];
                resultRestaurant = i;
            }
        }
        System.out.println("Minimum Distance for all people: " + globalMin);
        return resultRestaurant;
    }

}
