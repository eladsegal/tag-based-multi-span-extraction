import React from 'react';
import DatasetLocalSelector from './DatasetLocalSelector';
import DatasetListSelector from './DatasetListSelector';
import { shouldUpdate } from '../../../utils';
import {
    ListGroup,
    ListGroupItem,
    Input,
    FormGroup,
    Label
  } from 'reactstrap';

const updateSignals = ['dataset', 'useLocalDataset']
class DatasetLoader extends React.Component {
    constructor(props) {
        super(props);
        this.useLocalDatasetChange = this.useLocalDatasetChange.bind(this);
        this.datasetChange = this.datasetChange.bind(this);
        this.state = {
            dataset: undefined
        };
    }

    componentDidMount() {
        this.setState({ 
            useLocalDataset: this.props.useLocalDataset
        })
    }

    shouldComponentUpdate(nextProps, nextState) {
        if (this.state.useLocalDataset !== nextState.useLocalDataset) {
            this.setState({ dataset: undefined });
        }
        return shouldUpdate(updateSignals, this.props, this.state, nextProps, nextState)
    }

    componentDidUpdate(prevProps, prevState) {
        if (prevState.dataset !== this.state.dataset) {
            this.props.onDatasetChange(this.state.dataset);
        }
    }

    useLocalDatasetChange(e) {
        this.setState({ 
            useLocalDataset: e.currentTarget.getAttribute('use-local-dataset') !== true.toString()
        });
    }

    datasetChange(dataset) {
        this.setState({ dataset: dataset })
    }

    render() {
        return <ListGroup>
                <ListGroupItem>
                    <FormGroup check>
                        <Label check>
                            <Input type="checkbox" 
                            onChange={this.useLocalDatasetChange}
                            use-local-dataset={(this.state.useLocalDataset && 
                                this.state.useLocalDataset.toString()) || false.toString()} 
                            checked={this.state.useLocalDataset || false} 
                            />Use Local Dataset
                        </Label>
                    </FormGroup>
                </ListGroupItem>
                <ListGroupItem>
                    <DatasetSelector onChange={this.datasetChange} local={this.state.useLocalDataset}/>
                </ListGroupItem>
            </ListGroup>
    }
}

function DatasetSelector(props) {
    if (props.local) {
        return <DatasetLocalSelector onChange={props.onChange} />
    }
    return <DatasetListSelector onChange={props.onChange} />
}

export default DatasetLoader;
