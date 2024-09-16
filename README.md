
```
cd <ros workspace>/src
git clone https://github.com/khuechuong/culvert_sim
cd ..
catkin build culvert_sim 
```
Put [bigger_rough_3crack_2spall](https://github.com/khuechuong/culvert_sim/tree/main/model/bigger_rough_3crack_2spall) folder in ```.gazebo/model``` folder

```
roscd culvert_sim
cd model
mv bigger_rough_3crack_2spall ~/.gazebo/model/
```
POMDP Notes

Add an action called go to defect. 
Policy Model:
- This can only happen when Defect array > 0 and there exist a d in Defect that is not visited.
- Filter out the valid moves. like no hitting walls

Belief:
- stop updating after done
- update when further away is more of a sample update points around it, closer means updating more to the 
sample belief only on spots that is not visited (with some limitations) might be a bad idea


Future stuff:
- obstacles belief and filter out valid moves around obstacles
