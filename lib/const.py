class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


class ActionType:
    #  Pass, Move, Shoot, GetOn, GetOff, Occupy, ChangeState, RemoveKeep, JMPlan, GuideShoot, StopMove, WeaponLock, WeaponUnFold, CancelJMPlan = range(
    #      14)

    Pass, Move, Shoot, GetOn, GetOff, Occupy, ChangeState, RemoveKeep, JMPlan, GuideShoot, StopMove, WeaponLock, WeaponUnFold, CancelJMPlan, Ambush, Focus = range(
        16)


#    Ambush: Move to some place and stop moving for a while
#    Focus: All other valid units to focus attack
#    MoveS: Move to a 
class MoveType:
    Maneuver, March, Walk, Fly = range(4)


action_type_map = {ActionType.Shoot: 0}

RED, BLUE = 0, 1
scenario_bop = {
    201033019601:
        {
            RED:
                [0],
            BLUE:
                [1],
        },
    201033029601:
        {
            RED:
                [0, 100, 200],
            BLUE:
                [1, 101, 201],
        },
    201033039601:
        {
            RED:
                [0, 100, 200],
            BLUE:
                [1, 101, 201],
        }

}
