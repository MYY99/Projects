/* ========================================
 *
 * Copyright YOUR COMPANY, THE YEAR
 * All Rights Reserved
 * UNPUBLISHED, LICENSED SOFTWARE.
 *
 * CONFIDENTIAL AND PROPRIETARY INFORMATION
 * WHICH IS THE PROPERTY OF your company.
 *
 * ========================================
*/

#ifndef FUNCTIONS_SELF_DEFINED_H
#define FUNCTIONS_SELF_DEFINED_H
    
// general    
void move_straight(int mode);
void stop_movement();
void turn(int mode);

// state 2
int until_ultrasonic_detected(int *dist_reach_flag);

// state 3
void colour_sense(int mode, int* freq_leftPtr, int* freq_rightPtr);
int colour_determine(int red_freq, int blue_freq, int green_freq);

// state 4
void until_target_count(int target_count);
    
#endif

/* [] END OF FILE */
