import React from 'react';
import DatasetLoader from './dataset-loader/DatasetLoader';
import {
    ListGroup,
    ListGroupItem,
    CardGroup,
    Card,
    CardHeader,
    CardBody,
    Button
  } from 'reactstrap';
import { shouldUpdate } from '../../utils';
import AnswerTypeFilter from './AnswerTypeFilter';

const updateSignals = ['dataset', 'useLocalDataset', 'expandAll', 'filteredAnswerTypes']
class ExplorerSettings extends React.Component {
    constructor(props) {
        super(props);
        this.datasetChange = this.datasetChange.bind(this);
        this.filteredAnswerTypesChange = this.filteredAnswerTypesChange.bind(this);
        this.state = {
        };
    }

    componentDidMount() {
        this.setState({ 
            expandAll: this.props.expandAll,
            filteredAnswerTypes: this.props.filteredAnswerTypes
        })
    }

    shouldComponentUpdate(nextProps, nextState) {
        return shouldUpdate(updateSignals, this.props, this.state, nextProps, nextState)
    }

    componentDidUpdate(prevProps, prevState) {
        this.props.onChange(this.state);
    }

    datasetChange(dataset) {
        this.setState({dataset: dataset});
    }

    filteredAnswerTypesChange(filteredAnswerTypes) {
        this.setState({filteredAnswerTypes: filteredAnswerTypes});
    }

    render() {
        return <CardGroup>
                <Card>
                    <CardHeader>Dataset</CardHeader>
                    <CardBody>
                        <DatasetLoader onDatasetChange={this.datasetChange} useLocalDataset={this.props.useLocalDataset} />
                    </CardBody>
                </Card>
                <Card><CardBody>
                    <ListGroup>
                        <ListGroupItem>
                            <Button size='sm' onClick={() => this.setState({ 
                                    expandAll: !this.state.expandAll 
                                })}>
                                {this.state.expandAll ? "COLLAPSE AND UNFREEZE" : "EXPAND ALL AND FREEZE"}
                            </Button>
                        </ListGroupItem>
                        <ListGroupItem>
                            <Button size='sm' onClick={() => {
                                if (this.props.clearSelectedAnswersFunc) {
                                    this.props.clearSelectedAnswersFunc()
                                }
                            }}>CLEAR SELECTED ANSWERS
                            </Button>
                        </ListGroupItem>
                        <ListGroupItem>
                            {/*<FormGroup check>
                                <Label check>
                                    <Input type="checkbox" 
                                    onChange={this.useLocalDatasetChange}
                                    use-local-dataset={(this.state.useLocalDataset && 
                                        this.state.useLocalDataset.toString()) || false.toString()} 
                                    checked={this.state.useLocalDataset || false} 
                                    />Use Local Dataset
                                </Label>
                                    </FormGroup>*/}
                        </ListGroupItem>
                    </ListGroup>
                </CardBody></Card>
                <Card>
                    <CardHeader>Filter Answer Type</CardHeader>
                    <CardBody>
                        <AnswerTypeFilter onChange={this.filteredAnswerTypesChange} filteredAnswerTypes={this.props.filteredAnswerTypes} />
                    </CardBody>
                </Card>
        </CardGroup>
    }
}

export default ExplorerSettings;