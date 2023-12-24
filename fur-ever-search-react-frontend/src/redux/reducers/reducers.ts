import { combineReducers } from 'redux';

import HomeReducer from './home-reducer';

export const makeRootReducer = () => {
  return combineReducers({
    landings: HomeReducer,
  });
};

export default makeRootReducer
