
import { LOADING, MODE_TOGGLE, ADD_TO_CART, ADD_TO_LIKED_LIST, REMOVE_FROM_LIKED_LIST, REMOVE_FROM_CART } from '../constants';

const INITIAL_STATE = {
}

export const ACTION_HANDLERS: any = {

}

export default function AppReducer(state: any = INITIAL_STATE, action: any) : any {
    const handler = ACTION_HANDLERS[action.type]
    return handler ? handler(state, action) : state
}
