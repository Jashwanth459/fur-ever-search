import { applyMiddleware, compose, createStore } from 'redux';
import thunk from 'redux-thunk';

import makeRootReducer from '../reducers/reducers';

const store = (initialState = {}) => {
  const middleware = [thunk]
  const store = createStore(
    makeRootReducer(),
    initialState,
    compose(applyMiddleware(...middleware))
  )
  return store;
}

export default store
